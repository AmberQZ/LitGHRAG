#!/usr/bin/env python3
"""
Knowledge Graph Extraction Pipeline
Extracts entities, relations, and events from text data using transformer models.
"""
import re
import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
import json_repair
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.kg_construction.utils.json_processing.json_to_csv import json2csv
from atlas_rag.kg_construction.concept_generation import generate_concept
from atlas_rag.kg_construction.utils.csv_processing.merge_csv import merge_csv_files
from atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml import csvs_to_graphml, csvs_to_temp_graphml
from atlas_rag.kg_construction.concept_to_csv import all_concept_triples_csv_to_csv
from atlas_rag.kg_construction.utils.csv_processing.csv_add_numeric_id import add_csv_columns
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.vectorstore.create_neo4j_index import create_faiss_index
from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS
from atlas_rag.kg_construction.triple_config import ProcessingConfig
# Constants
TOKEN_LIMIT = 1024
INSTRUCTION_TOKEN_ESTIMATE = 200
CHAR_TO_TOKEN_RATIO = 3.5



class TextChunker:
    """Handles text chunking based on token limits."""
    
    def __init__(self, max_tokens: int = TOKEN_LIMIT, instruction_tokens: int = INSTRUCTION_TOKEN_ESTIMATE):
        self.max_tokens = max_tokens
        self.instruction_tokens = instruction_tokens
        self.char_ratio = CHAR_TO_TOKEN_RATIO
        
    def calculate_max_chars(self) -> int:
        """Calculate maximum characters per chunk."""
        available_tokens = self.max_tokens - self.instruction_tokens
        return int(available_tokens * self.char_ratio)
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limits."""
        max_chars = self.calculate_max_chars()
        chunks = []
        
        while len(text) > max_chars:
            chunks.append(text[:max_chars])
            text = text[max_chars:]
        
        if text:  # Add remaining text
            chunks.append(text)
            
        return chunks


class DatasetProcessor:
    """Processes and prepares dataset for knowledge graph extraction."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.chunker = TextChunker()
        
    def filter_language_content(self, sample: Dict[str, Any]) -> bool:
        """Check if content is in English."""
        metadata = sample.get("metadata", {})
        language = metadata.get("lang", "en")  # Default to English if not specified
        supported_languages = list(TRIPLE_INSTRUCTIONS.keys())
        return language in supported_languages
    
    
    def create_sample_chunks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from a single sample."""
        original_text = sample.get("text", "")
        if self.config.remove_doc_spaces:
            original_text = re.sub(r'\s+', ' ',original_text).strip()
        text_chunks = self.chunker.split_text(original_text)
        chunks = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                "id": sample["id"],
                "text": chunk_text,
                "chunk_id": chunk_idx,
                "metadata": sample["metadata"]
            }
            chunks.append(chunk_data)
            
        return chunks
    
    def prepare_dataset(self, raw_dataset) -> List[Dict[str, Any]]:
        """Process raw dataset into chunks suitable for processing with generalized slicing."""
        
        processed_samples = []
        total_texts = len(raw_dataset)
        
        # Handle edge cases
        if total_texts == 0:
            print(f"No texts found for shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple}")
            return processed_samples
        
        # Calculate base and remainder for fair distribution
        base_texts_per_shard = total_texts // self.config.total_shards_triple
        remainder = total_texts % self.config.total_shards_triple
        
        # Calculate start index
        if self.config.current_shard_triple < remainder:
            start_idx = self.config.current_shard_triple * (base_texts_per_shard + 1)
        else:
            start_idx = remainder * (base_texts_per_shard + 1) + (self.config.current_shard_triple - remainder) * base_texts_per_shard
        
        # Calculate end index
        if self.config.current_shard_triple < remainder:
            end_idx = start_idx + (base_texts_per_shard + 1)
        else:
            end_idx = start_idx + base_texts_per_shard
        
        # Ensure indices are within bounds
        start_idx = min(start_idx, total_texts)
        end_idx = min(end_idx, total_texts)
        
        print(f"Processing shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple} "
            f"(texts {start_idx}-{end_idx-1} of {total_texts}, {end_idx - start_idx} documents)")
        
        # Process documents in assigned shard
        for idx in range(start_idx, end_idx):
            sample = raw_dataset[idx]
            
            # Filter by language
            if not self.filter_language_content(sample):
                print(f"Unsupported language in sample {idx}, skipping.")
                continue
                
            # Create chunks
            chunks = self.create_sample_chunks(sample)
            processed_samples.extend(chunks)
            
            # Debug mode early termination
            if self.config.debug_mode and len(processed_samples) >= 20:
                print("Debug mode: Stopping at 20 chunks")
                break
        
        print(f"Generated {len(processed_samples)} chunks for shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple}")
        return processed_samples


class CustomDataLoader:
    """Custom data loader for knowledge graph extraction."""
    
    def __init__(self, dataset, processor: DatasetProcessor):
        self.raw_dataset = dataset
        self.processor = processor
        self.processed_data = processor.prepare_dataset(dataset)
        self.stage_to_prompt_dict = {
            "stage_1": "entity_relation",
            "stage_2": "event_entity",
            "stage_3": "event_relation"
        }
        
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def create_batch_instructions(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        messages_dict = {
            'stage_1': [],
            'stage_2': [],
            'stage_3': []
        }
        for item in batch_data:
            # get language
            language = item.get("metadata",{}).get("lang", "en")
            system_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['system'] 
            stage_1_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['entity_relation'] + TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['passage_start'] + '\n' + item["text"]
            stage_2_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['event_entity'] + TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['passage_start'] + '\n'+ item["text"]
            stage_3_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['event_relation'] + TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['passage_start'] + '\n'+ item["text"]
            stage_one_message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": stage_1_msg}
            ]
            stage_two_message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": stage_2_msg}
            ]
            stage_three_message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": stage_3_msg}
            ]
            messages_dict['stage_1'].append(stage_one_message)
            messages_dict['stage_2'].append(stage_two_message)
            messages_dict['stage_3'].append(stage_three_message)
        
        return messages_dict
    
    def __iter__(self):
        """Iterate through batches."""
        batch_size = self.processor.config.batch_size_triple
        start_idx = self.processor.config.resume_from * batch_size
        
        for i in tqdm(range(start_idx, len(self.processed_data), batch_size)):
            batch_data = self.processed_data[i:i + batch_size]
            
            # Prepare instructions
            instructions = self.create_batch_instructions(batch_data)
            
            # Extract batch information
            batch_ids = [item["id"] for item in batch_data]
            batch_metadata = [item["metadata"] for item in batch_data]
            batch_texts = [item["text"] for item in batch_data]
            
            yield instructions, batch_ids, batch_texts, batch_metadata


class OutputParser:
    """Parses model outputs and extracts structured data."""
    def __init__(self):
        pass

    def extract_structured_data(self, outputs: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract structured data from model outputs."""
        results = []
        for output in outputs:
            parsed_data = json_repair.loads(output)
            results.append(parsed_data)
            
        return results


class KnowledgeGraphExtractor:
    """Main class for knowledge graph extraction pipeline."""
    
    def __init__(self, model:LLMGenerator, config: ProcessingConfig):
        self.config = config
        self.model = None
        self.parser = None
        self.model = model
        self.model_name = model.model_name
        self.parser = OutputParser()
    
    def load_dataset(self) -> Any:
        """Load and prepare dataset."""
        data_path = Path(self.config.data_directory)
        all_files = os.listdir(data_path)
        
        valid_files = [
            filename for filename in all_files
            if filename.startswith(self.config.filename_pattern) and
            (filename.endswith(".json.gz") or filename.endswith(".json") or filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"))
        ]
        
        print(f"Found data files: {valid_files}")
        data_files = valid_files
        dataset_config = {"train": data_files}
        return load_dataset(self.config.data_directory, data_files=dataset_config["train"])
    
    def process_stage(self, instructions: Dict[str, str], stage = 1) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """Process first stage: entity-relation extraction."""
        outputs = self.model.triple_extraction(messages=instructions, max_tokens=self.config.max_new_tokens, stage=stage, record=self.config.record)
        if self.config.record:
            text_outputs = [output[0] for output in outputs]
        else:
            text_outputs = outputs
        structured_data = self.parser.extract_structured_data(text_outputs)
        return outputs, structured_data
    
    def create_output_filename(self) -> str:
        """Create output filename with timestamp and shard info."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name_safe = self.config.model_path.replace("/", "_")
        
        filename = (f"{model_name_safe}_{self.config.filename_pattern}_output_"
                   f"{timestamp}_{self.config.current_shard_triple + 1}_in_{self.config.total_shards_triple}.json")
        
        extraction_dir = os.path.join(self.config.output_directory, "kg_extraction")
        os.makedirs(extraction_dir, exist_ok=True)
        
        return os.path.join(extraction_dir, filename)
    
    def prepare_result_dict(self, batch_data: Tuple, stage_outputs: Tuple, index: int) -> Dict[str, Any]:
        """Prepare result dictionary for a single sample."""
        ids, original_texts, metadata = batch_data
        (stage1_results, entity_relations), (stage2_results, event_entities), (stage3_results, event_relations) = stage_outputs
        if self.config.record:
            stage1_outputs = [output[0] for output in stage1_results]
            stage1_usage = [output[1] for output in stage1_results]
            stage2_outputs = [output[0] for output in stage2_results]
            stage2_usage = [output[1] for output in stage2_results]
            stage3_outputs = [output[0] for output in stage3_results]
            stage3_usage = [output[1] for output in stage3_results]
        else:
            stage1_outputs = stage1_results
            stage2_outputs = stage2_results
            stage3_outputs = stage3_results
        result = {
            "id": ids[index],
            "metadata": metadata[index],
            "original_text": original_texts[index],
            "entity_relation_dict": entity_relations[index],
            "event_entity_relation_dict": event_entities[index],
            "event_relation_dict": event_relations[index],
            "output_stage_one": stage1_outputs[index],
            "output_stage_two": stage2_outputs[index],
            "output_stage_three": stage3_outputs[index],
        }
        if self.config.record:
            result['usage_stage_one'] = stage1_usage[index]
            result['usage_stage_two'] = stage2_usage[index]
            result['usage_stage_three'] = stage3_usage[index]
        
        # Handle date serialization
        if 'date_download' in result['metadata']:
            result['metadata']['date_download'] = str(result['metadata']['date_download'])
            
        return result
    
    def debug_print_result(self, result: Dict[str, Any]):
        """Print result for debugging."""
        for key, value in result.items():
            print(f"{key}: {value}")
            print("-" * 100)
    
    def run_extraction(self):
        """Run the complete knowledge graph extraction pipeline."""
        # Setup
        os.makedirs(self.config.output_directory+'/kg_extraction', exist_ok=True)
        dataset = self.load_dataset()
        
        if self.config.debug_mode:
            print("Debug mode: Processing only 20 samples")
        
        # Create data processor and loader
        processor = DatasetProcessor(self.config)
        data_loader = CustomDataLoader(dataset["train"], processor)
        
        output_file = self.create_output_filename()
        print(f"Model: {self.config.model_path}")
        
        batch_counter = 0
        
        with torch.no_grad():
            with open(output_file, "w") as output_stream:
                for batch in data_loader:
                    batch_counter += 1
                    messages_dict, batch_ids, batch_texts, batch_metadata = batch
                    
                    # Process all three stages
                    stage1_results = self.process_stage(messages_dict['stage_1'],1)
                    stage2_results = self.process_stage(messages_dict['stage_2'],2)
                    stage3_results = self.process_stage(messages_dict['stage_3'],3)
                    
                    # Combine results
                    batch_data = (batch_ids, batch_texts, batch_metadata)
                    stage_outputs = (stage1_results, stage2_results, stage3_results)

                    # Write results
                    print(f"Processed {batch_counter} batches ({batch_counter * self.config.batch_size_triple} chunks)")
                    for i in range(len(batch_ids)):
                        result = self.prepare_result_dict(batch_data, stage_outputs, i)
                        
                        if self.config.debug_mode:
                            self.debug_print_result(result)
   
                        output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_stream.flush()

    def convert_json_to_csv(self):
        json2csv(
            dataset = self.config.filename_pattern, 
            output_dir=f"{self.config.output_directory}/triples_csv",
            data_dir=f"{self.config.output_directory}/kg_extraction"
        )
        csvs_to_temp_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            triple_edge_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv",
            config = self.config
        )
    
    def generate_concept_csv_temp(self, batch_size: int = None, **kwargs):
        generate_concept(
            model=self.model,
            input_file=f"{self.config.output_directory}/triples_csv/missing_concepts_{self.config.filename_pattern}_from_json.csv",
            output_folder=f"{self.config.output_directory}/concepts",
            output_file="concept.json",
            logging_file=f"{self.config.output_directory}/concepts/logging.txt",
            config=self.config,
            batch_size=batch_size if batch_size else self.config.batch_size_concept,
            shard=self.config.current_shard_concept,
            num_shards=self.config.total_shards_concept,
            record = self.config.record,
            **kwargs
        )
    
    def create_concept_csv(self):
        merge_csv_files(
            output_file=f"{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv",
            input_dir=f"{self.config.output_directory}/concepts",
        )
        all_concept_triples_csv_to_csv(
            node_file=f'{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv',
            edge_file=f'{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv',
            concepts_file=f'{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv',
            output_node_file=f'{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_edge_file=f'{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_full_concept_triple_edges=f'{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
        )
        
    def convert_to_graphml(self):
        csvs_to_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            text_node_file=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            concept_node_file=f"{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv",
            triple_edge_file=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            text_edge_file=f"{self.config.output_directory}/triples_csv/text_edges_{self.config.filename_pattern}_from_json.csv",
            concept_edge_file=f"{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            output_file=f"{self.config.output_directory}/kg_graphml/{self.config.filename_pattern}_graph.graphml",
        )
    
    def add_numeric_id(self):
        add_csv_columns(
            node_csv=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            edge_csv=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            text_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            node_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            edge_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            text_with_numeric_id=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_numeric_id.csv",
        )

    def compute_kg_embedding(self, encoder_model:BaseEmbeddingModel, batch_size: int = 2048):
        encoder_model.compute_kg_embedding(
            node_csv_without_emb=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            node_csv_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            edge_csv_without_emb=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            edge_csv_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.csv",
            text_node_csv_without_emb=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            text_node_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            batch_size = 2048
        )

    def create_faiss_index(self, index_type="HNSW,Flat", faiss_gpu=True):
        create_faiss_index(self.config.output_directory, self.config.filename_pattern, index_type, faiss_gpu)

def parse_command_line_arguments() -> ProcessingConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Pipeline")
    
    parser.add_argument("-m", "--model", type=str, required=True,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model path for knowledge extraction")
    parser.add_argument("--data_dir", type=str, default="your_data_dir",
                       help="Directory containing input data")
    parser.add_argument("--file_name", type=str, default="en_simple_wiki_v0",
                       help="Filename pattern to match")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="./generation_result_debug",
                       help="Output directory for results")
    parser.add_argument("--total_shards_triple", type=int, default=1,
                       help="Total number of data shards")
    parser.add_argument("--shard", type=int, default=0,
                       help="Current shard index")
    parser.add_argument("--bit8", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--resume", type=int, default=0,
                       help="Resume from specific batch")
    
    args = parser.parse_args()
    
    return ProcessingConfig(
        model_path=args.model,
        data_directory=args.data_dir,
        filename_pattern=args.file_name,
        batch_size=args.batch_size,
        output_directory=args.output_dir,
        total_shards_triple=args.total_shards_triple,
        current_shard_triple=args.shard,
        use_8bit=args.bit8,
        debug_mode=args.debug,
        resume_from=args.resume
    )


def main():
    """Main entry point for the knowledge graph extraction pipeline."""
    config = parse_command_line_arguments()
    extractor = KnowledgeGraphExtractor(config)
    extractor.run_extraction()


if __name__ == "__main__":
    main() 