import os
import json
import numpy as np
from logging import Logger
from litgh_rag.retrieval.retriever import BaseRetriever, BaseEdgeRetriever, BasePassageRetriever
from typing import List
from datetime import datetime
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from litgh_rag.retrieval.embedding_model import NvEmbed, SentenceEmbedding
from litgh_rag.reader.llm_generator import LLMGenerator
from litgh_rag.evaluation.evaluation import QAJudger
from dataclasses import dataclass
from transformers import BertTokenizer, BertForSequenceClassification


def normalize_embeddings(embeddings):
    """Normalize the embeddings to unit length (L2 norm)."""
    if isinstance(embeddings, torch.Tensor):
        # Handle PyTorch tensors
        norm_emb = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        # Handle numpy arrays
        norm_emb = F.normalize(torch.tensor(embeddings), p=2, dim=1).detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(embeddings)}. Must be torch.Tensor or np.ndarray")
    
    return norm_emb

@dataclass
class BenchMarkConfig:
    """
    Configuration class for benchmarking.

    Attributes:
        dataset_name (str): Name of the dataset. Default is "hotpotqa".
        question_file (str): Path to the question file. Default is "hotpotqa".
        graph_file (str): Path to the graph file. Default is "hotpotqa_concept.graphml".
        include_events (bool): Whether to include events. Default is False.
        include_concept (bool): Whether to include concepts. Default is False.
        reader_model_name (str): Name of the reader model. Default is "meta-llama/Llama-2-7b-chat-hf".
        encoder_model_name (str): Name of the encoder model. Default is "nvidia/NV-Embed-v2".
        number_of_samples (int): Number of samples to use from the dataset. Default is -1 (use all samples).
    """
    dataset_name: str = "hotpotqa"
    question_file: str = "hotpotqa"
    include_events: bool = False
    include_concept: bool = False
    reader_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    encoder_model_name: str = "nvidia/NV-Embed-v2"
    number_of_samples: int = -1  # Default to -1 to use all samples
    react_max_iterations: int = 5
        

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


class RAGBenchmark:
    def __init__(self, config:BenchMarkConfig, logger:Logger = None):
        self.config = config
        self.logger = logger
        self.logging : bool = self.logger is not None

    def load_encoder_model(self, encoder_model_name, **kwargs):
        if encoder_model_name == "nvidia/NV-Embed-v2":
            sentence_encoder = AutoModel.from_pretrained("nvidia/NV-Embed-v2", **kwargs)
            return NvEmbed(sentence_encoder)
        else:
            sentence_encoder = SentenceTransformer(encoder_model_name, **kwargs)
            return SentenceEmbedding(sentence_encoder)

    def run(self, retrievers:List[BaseRetriever], 
                  llm_generator:LLMGenerator,
                  ):
        qa_judge = QAJudger()
        result_list = []
        with open(self.config.question_file, "r") as f:
            data = json.load(f)
            print(f"Data loaded from {self.config.question_file}")
        if self.config.number_of_samples > 0:
            data = data[:self.config.number_of_samples]
            print(f"Using only the first {self.config.number_of_samples} samples from the dataset")

        for idx, sample in enumerate(tqdm(data)):
            # print(f"Processing sample idx={idx}: {sample.get('question', '')[:50]}")
            question = sample["question"]

            answer = None
            if 'answer' in sample or 'gold_ans' in sample:
                answer = sample['answer'] if 'answer' in sample else sample['gold_ans']
            elif 'reference' in sample:
                answer = sample['reference']
            elif 'obj' in sample:
                answer = set(
                    [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
                answer = list(answer)
            assert answer is not None
            if isinstance(answer, str):
                answer = [answer]

            gold_file_ids = []
            if self.config.dataset_name in ("hotpotqa", "2wikimultihopqa"):
                for fact in sample["supporting_facts"]:
                    gold_file_ids.append(fact[0])
            elif self.config.dataset_name == "musique":
                for paragraph in sample["paragraphs"]:
                    if paragraph["is_supporting"]:
                        gold_file_ids.append(paragraph["paragraph_text"])
            elif self.config.dataset_name == "popqa":
                # Handle your specific dataset case here
                    gold_paragraphs = []
                    for item in sample['paragraphs']:
                        if 'is_supporting' in item and item['is_supporting'] is False:
                            continue
                        gold_paragraphs.append(item)
                    gold_file_ids = [(item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]
            elif self.config.dataset_name == "nq_rear":
                gold_file_ids = [ item['text'] for item in sample['contexts'] if item['is_supporting']]
            else:
                print(f"Warning: Dataset {self.config.dataset_name} not specifically handled for gold_file_ids extraction.")
                
            result = {
                "question": question,
                "answer": answer,
                "gold_file_ids": gold_file_ids,
            } 
            

            if self.logging:
                self.logger.info(f"Question: {question}")
            for retriever in retrievers:
                # print(f"  Using retriever: {retriever.__class__.__name__}")
                # 问题分类
                MODEL_PATH = "./trained_bert_all5_question_classifier"

                # 加载模型和分词器
                model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model.eval()
                if torch.cuda.is_available():
                    model.cuda()
                inputs = tokenizer(question, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
                inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = outputs.logits.argmax(dim=-1).item()
                if pred == 0: #简单问题
                    vner=3
                    vq2node=10
                    vq2edge=20
                    topN=3
                else: #多跳问题
                    vner=6
                    vq2node=25
                    vq2edge=60
                    topN=5

                sorted_context, sorted_context_ids, triples, events = retriever.retrieve(question, topN=topN, vner=vner, vq2node=vq2node, vq2edge=vq2edge)
                    
                if isinstance(retriever, BasePassageRetriever):
                    if pred == 0:  # 简单问题
                    # 4.1 qwen only passages no think
                      retrieved_context = ". ".join(sorted_context)
                      llm_generated_answer = llm_generator.generate_with_context(question, retrieved_context, max_new_tokens=2048, temperature=0.5, think=False)
                      if isinstance(llm_generated_answer, str) and "</think>" in llm_generated_answer:
                          llm_generated_answer = llm_generated_answer.split("</think>", 1)[1].replace("\n", "").strip()
                    else:
                    # 4.2 qwen think 所有信息                   
                      filter_result = llm_generator.filter_with_triples_event(question, events, triples, max_new_tokens=2048, temperature=0.5)
                      if isinstance(filter_result, str) and "</think>" in filter_result:
                          filter_result = filter_result.split("</think>", 1)[1].replace("\n", "").strip()
                      import re
                      if isinstance(filter_result, str):
                          cleaned_filter_result = re.sub(r"\s+", " ", filter_result.replace("'", "")).strip()
                      else:
                          cleaned_filter_result = re.sub(r"\s+", " ", str(filter_result).replace("'", "")).strip()
                      retrieved_context = "Passages: " + "\n".join(sorted_context) + "\n" + cleaned_filter_result

                      llm_generated_answer = llm_generator.generate_with_pass_te(question, retrieved_context, max_new_tokens=2048, temperature=0.5)
                      if isinstance(llm_generated_answer, str) and "</think>" in llm_generated_answer:
                          llm_generated_answer = llm_generated_answer.split("</think>", 1)[1].replace("\n", "").strip()
                                      

                if self.logging:
                    self.logger.info(f"{retriever.__class__.__name__} retrieved passages: {sorted_context}")
                    self.logger.info(f"{retriever.__class__.__name__} generated answer: {llm_generated_answer}")

                short_answer = qa_judge.split_answer(llm_generated_answer)
                # print(short_answer)
                em, f1 = qa_judge.judge_simple_and_multi(short_answer, answer)
                result[f"{retriever.__class__.__name__ }_em"] = em
                result[f"{retriever.__class__.__name__ }_f1"] = f1
                result[f"{retriever.__class__.__name__ }_passages"] = sorted_context

                result[f"{retriever.__class__.__name__ }_id"] = sorted_context_ids
                result[f"{retriever.__class__.__name__ }_generated_answer"] = llm_generated_answer
                result[f"{retriever.__class__.__name__ }short_answer"] = short_answer
                

                if self.config.dataset_name in ("hotpotqa", "2wikimultihopqa"):
                    recall_2, recall_5 = qa_judge.recall(sorted_context_ids, gold_file_ids)
                else:
                    
                    recall_2, recall_5 = qa_judge.recall(sorted_context, gold_file_ids)
                    
                result[f"{retriever.__class__.__name__ }_recall@2"] = recall_2
                result[f"{retriever.__class__.__name__ }_recall@5"] = recall_5
                
            result_list.append(result)

        self.save_results(result_list, [retriever.__class__.__name__ for retriever in retrievers])
        return result_list, [retriever.__class__.__name__ for retriever in retrievers]
    

    def save_results(self, result_list, retriever_names:List[str]):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        
        dataset_name = self.config.dataset_name
        include_events = self.config.include_events
        include_concept = self.config.include_concept
        encoder_model_name = self.config.encoder_model_name
        reader_model_name = self.config.reader_model_name
        
        # use last part of model name as identifier
        if "/" in encoder_model_name:
            encoder_model_name = encoder_model_name.split("/")[-1]
        if "/" in reader_model_name:
            reader_model_name = reader_model_name.split("/")[-1]

        summary_file = f"./result/{dataset_name}/summary_{formatted_time}_event{include_events}_concept{include_concept}_{encoder_model_name}_{reader_model_name}.json"
        if not os.path.exists(os.path.dirname(summary_file)):
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        result_dir = f"./result/{dataset_name}/result_{formatted_time}_event{include_events}_concept{include_concept}_{encoder_model_name}_{reader_model_name}.json"
        if not os.path.exists(os.path.dirname(result_dir)):
            os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        
        summary_dict = self.calculate_summary(result_list, retriever_names)
        
        with open(summary_file, "w") as f_summary:
            json.dump(summary_dict, f_summary)
            f_summary.write("\n")

        with open(result_dir, "w") as f:
            for result in result_list:
                json.dump(result, f)
                f.write("\n")
    
    def calculate_summary(self, result_list, method):
        summary_dict = {}
        for retriever_name in method:
            if not all(f"{retriever_name}_em" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_em in results")
            if not all(f"{retriever_name}_f1" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_f1 in results")
            
            average_em = sum([result[f"{retriever_name}_em"] for result in result_list]) / len(result_list)
            average_f1 = sum([result[f"{retriever_name}_f1"] for result in result_list]) / len(result_list)
            
            # Only calculate recall metrics if they exist in the results
            if all(f"{retriever_name}_recall@2" in result for result in result_list):
                average_recall_2 = sum([result[f"{retriever_name}_recall@2"] for result in result_list]) / len(result_list)
                average_recall_5 = sum([result[f"{retriever_name}_recall@5"] for result in result_list]) / len(result_list)
                summary_dict.update({
                    f"{retriever_name}_average_f1": average_f1,
                    f"{retriever_name}_average_em": average_em,
                    f"{retriever_name}_average_recall@2": average_recall_2,
                    f"{retriever_name}_average_recall@5": average_recall_5,
                })
            else:
                # For ReAct mode where recall metrics don't exist
                summary_dict.update({
                    f"{retriever_name}_average_f1": average_f1,
                    f"{retriever_name}_average_em": average_em,
                })
                
        return summary_dict
