from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from litgh_rag.reader import LLMGenerator
from openai import OpenAI
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, default='hotpotqa', help='Dataset keyword')
    args = parser.parse_args()
    keyword = args.keyword

    model_name = "llama3.1:8b"
    client = OpenAI(
            base_url="http://localhost:11434/v1",  # 本地 Ollama API 地址
            api_key="ollama"  # 随便写，不验证
        )

    triple_generator = LLMGenerator(client, model_name=model_name,)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    output_directory = f'./dataset/{keyword}/{model_name}'
    kg_extraction_config = ProcessingConfig(
            model_path=model_name,
            data_directory="./dataset",
            filename_pattern=keyword,
            batch_size_triple=4,
            batch_size_concept=64,
            output_directory=f"{output_directory}",
            record=True,
            max_new_tokens=4096,
            max_workers=8,
            remove_doc_spaces=True,
        )

    kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

    # Construct entity&event graph
    kg_extractor.run_extraction() # Involved LLM Generation
    # Convert Triples Json to CSV
    kg_extractor.convert_json_to_csv()
    # Concept Generation
    kg_extractor.generate_concept_csv_temp(language='en')
    # Create Concept CSV
    kg_extractor.create_concept_csv()
    # Convert csv to graphml for networkx
    kg_extractor.convert_to_graphml()


if __name__ == "__main__":
    main()