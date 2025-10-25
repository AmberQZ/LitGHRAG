import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from configparser import ConfigParser
from openai import OpenAI
from litgh_rag.retrieval import  LitGHRAGRetriever
from atlas_rag.llm_generator import LLMGenerator
from litgh_rag.retrieval import NvEmbed, SentenceEmbedding
from sentence_transformers import SentenceTransformer
from litgh_rag.evaluation import BenchMarkConfig, RAGBenchmark
from litgh_rag import create_embeddings_and_index, setup_logger
from transformers import AutoModel
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, default='2wikimultihopqa', help='Dataset keyword')
    args = parser.parse_args()
    keyword = args.keyword

    encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_model = SentenceTransformer(encoder_model_name, trust_remote_code=True, model_kwargs={'device_map': "auto"})
    sentence_encoder = SentenceEmbedding(sentence_model)
    reader_model_name = "qwen3:4b"
    client = OpenAI(
        base_url="http://localhost:11434/v1",  # 本地 Ollama API 地址
        api_key="ollama"  # 随便写，不验证
    )
    llm_generator = LLMGenerator(client=client, model_name=reader_model_name)

    # Create embeddings and index
    working_directory = f'{keyword}'
    data = create_embeddings_and_index(
        sentence_encoder=sentence_encoder,
        model_name = encoder_model_name,
        working_directory=working_directory,
        keyword=keyword,
        include_concept=True,
        include_events=True,
        normalize_embeddings=True,
        text_batch_size=64,
        node_and_edge_batch_size=64,
    )

    # Configure benchmarking
    benchmark_config = BenchMarkConfig(
        dataset_name=keyword,
        question_file=f"benchmark_data/{keyword}.json",
        include_concept=True,                                      
        include_events=True,
        reader_model_name=reader_model_name,
        encoder_model_name=encoder_model_name,
        number_of_samples=-1,  # -1 for all samples
    )

    # Set up logger
    logger = setup_logger(benchmark_config)

    # Initialize LitGHRAGRetriever
    LitGHRAG_retriever = LitGHRAGRetriever(
        llm_generator=llm_generator,
        sentence_encoder=sentence_encoder,
        data=data,
        logger=logger
    )

    benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
    result_list, retriever_names = benchmark.run([LitGHRAG_retriever], llm_generator=llm_generator)
    summary = benchmark.calculate_summary(result_list, retriever_names)
    print(f"===={reader_model_name} {keyword}====================")
    print(summary)

if __name__ == "__main__":
    main()