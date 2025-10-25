import argparse
import os
import json
import networkx as nx
from tqdm import tqdm
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
import os
import datetime
from litgh_rag.evaluation.benchmark import BenchMarkConfig


def get_node_id(text):
    # Implement your node ID generation logic here
    return hash(text.strip().lower())

def clean_text(text):
    # Implement your text cleaning logic here
    return text.strip()

def process_triples(triples, graph, original_text_node, file_id, node_type):
    """Helper function to process entity/event triples and add to graph"""
    for triple in triples:
        head, rel, tail = triple
        head_id = get_node_id(head)
        tail_id = get_node_id(tail)
        
        # Add nodes
        graph.add_node(head_id, type=node_type, id=head)
        graph.add_node(tail_id, type=node_type, id=tail)
        
        # Add mention edges
        graph.add_edge(head_id, original_text_node, relation='mention in')
        graph.add_edge(tail_id, original_text_node, relation='mention in')
        
        # Add main relation edge
        graph.add_edge(head_id, tail_id, relation=rel)
        
        # Update file IDs for nodes and edges
        for node_id in [head_id, tail_id]:
            if "file_id" not in graph.nodes[node_id]:
                graph.nodes[node_id]["file_id"] = str(file_id)
            else:
                # Append regardless of existing values (maintains original behavior)
                graph.nodes[node_id]["file_id"] += "," + str(file_id)

        # Update file IDs for edges (was missing in optimized version)
        edge = graph.edges[head_id, tail_id]
        if "file_id" not in edge:
            edge["file_id"] = str(file_id)
        else:
            edge["file_id"] += "," + str(file_id)

def process_keyword(keyword, triple_input_dir, passage_input_dir, output_dir):
    """Process all files for a given keyword and build knowledge graph"""
    # Get relevant files
    triple_files = [f for f in os.listdir(triple_input_dir) if keyword in f]
    passage_files = [f for f in os.listdir(passage_input_dir) if keyword in f]

    if not passage_files:
        print(f"No passage files found for keyword: {keyword}")
        return
    if not triple_files:
        print(f"No triple files found for keyword: {keyword}")
        return

    # Initialize graph
    G = nx.DiGraph()
    passage_path = os.path.join(passage_input_dir, passage_files[0])

    # Add passage nodes
    with open(passage_path) as f:
        for line in tqdm(f, desc="Adding passages"):
            data = json.loads(line)
            text = data["text"].strip()
            if not text or text.isspace():
                continue
            
            node_id = get_node_id(text)
            G.add_node(node_id, type="passage", id=text, file_id=data["id"])

    # Process triple files
    for triple_file in tqdm(triple_files, desc=f"Processing {keyword} files"):
        file_path = os.path.join(triple_input_dir, triple_file)
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                original_text = data["original_text"]
                file_id = data["id"]
                original_text_node = get_node_id(original_text)

                # Process entity relations
                entity_triples = [
                    (clean_text(t["Head"]), clean_text(t["Relation"]), clean_text(t["Tail"]))
                    for t in data["entity_relation_dict"]
                    if all(k in t for k in ["Head", "Relation", "Tail"])
                ]
                process_triples(entity_triples, G, original_text_node, file_id, "entity")

                # Process event relations
                event_triples = [
                    (clean_text(t["Head"]), clean_text(t["Relation"]), clean_text(t["Tail"]))
                    for t in data["event_relation_dict"]
                    if all(k in t for k in ["Head", "Relation", "Tail"])
                ]
                process_triples(event_triples, G, original_text_node, file_id, "event")

                # Process event-entity relations
                event_entity_triples = []
                for event in data["event_entity_relation_dict"]:
                    if "Event" not in event or "Entity" not in event:
                        continue
                    evt = clean_text(event["Event"])
                    for ent in event["Entity"]:
                        ent_clean = clean_text(ent)
                        if evt and ent_clean:
                            event_entity_triples.append((evt, "is participated by", ent_clean))
                process_triples(event_entity_triples, G, original_text_node, file_id, "event_entity")

    # Save graph
    output_path = os.path.join(output_dir, f"{keyword}_kg_from_corpus.graphml")
    nx.write_graphml(G, output_path, infer_numeric_types=True)
    print(f"Saved graph for {keyword} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


def setup_logger(config:BenchMarkConfig) -> Logger:
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    log_file_path = f'./log/{config.dataset_name}_event{config.include_events}_concept{config.include_concept}_{date_time}.log'
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.INFO)
    max_bytes = 50 * 1024 * 1024  # 50 MB
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
    logger.addHandler(handler)
    
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to knowledge graphs")
    parser.add_argument("--triple_input_dir", type=str, default="./processed_data/output_kg_triples")
    parser.add_argument("--passage_input_dir", type=str, default="./processed_data/input_for_kg_construction")
    parser.add_argument("--output_dir", type=str, default="./processed_data/kg_graphml")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each keyword
    for keyword in ["2wikimultihopqa", "hotpotqa", "musique"]:
        process_keyword(
            keyword=keyword,
            triple_input_dir=args.triple_input_dir,
            passage_input_dir=args.passage_input_dir,
            output_dir=args.output_dir
        )