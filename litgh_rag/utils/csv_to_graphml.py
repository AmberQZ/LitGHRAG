import networkx as nx
import csv
import ast
import hashlib
import os

def get_node_id(entity_name, entity_to_id={}):
    """Returns existing or creates new nX ID for an entity using a hash-based approach."""
    if entity_name not in entity_to_id:
        # Use a hash function to generate a unique ID
        hash_object = hashlib.sha256(entity_name.encode('utf-8'))
        hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash
        # Use the first 8 characters of the hash as the ID (you can adjust the length as needed)
        entity_to_id[entity_name] = hash_hex
    return entity_to_id[entity_name]

def csvs_to_temp_graphml(triple_node_file, triple_edge_file):
    '''
    Convert triples CSV files into a networkx graph, for conceptualization context sampling
    - Triple nodes: Nodes representing triples, with properties like subject, predicate, object.
    - Triple edges: Edges representing relationships between triples, with properties like relation type.
    
    DiGraph networkx attributes:
    Node:
    - type: Type of the node (e.g., entity, event, text).
    - file_id: List of text IDs the node is associated with.
    - id: Node Name 
    Edge:
    - relation: relation name
    - file_id: List of text IDs the edge is associated with.
    - type: Type of the edge (e.g., Source, Relation).
    - synsets: List of synsets associated with the edge.
    
    '''
    g = nx.DiGraph()
    entity_to_id = {}

    # Add triple nodes
    with open(triple_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            g.add_node(get_node_id(node_id, entity_to_id), id=node_id, type=row["type"])

    # Add triple edges
    with open(triple_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])

    return g
    

def csvs_to_graphml(triple_node_file, text_node_file, concept_node_file,
                    triple_edge_file, text_edge_file, concept_edge_file,
                    output_file):
    '''
    Convert multiple CSV files into a single GraphML file.
    
    Types of nodes to be added to the graph:
    - Triple nodes: Nodes representing triples, with properties like subject, predicate, object.
    - Text nodes: Nodes representing text, with properties like text content.
    - Concept nodes: Nodes representing concepts, with properties like concept name and type.

    Types of edges to be added to the graph:
    - Triple edges: Edges representing relationships between triples, with properties like relation type.
    - Text edges: Edges representing relationships between text and nodes, with properties like text type.
    - Concept edges: Edges representing relationships between concepts and nodes, with properties like concept type.
    
    DiGraph networkx attributes:
    Node:
    - type: Type of the node (e.g., entity, event, text, concept).
    - file_id: List of text IDs the node is associated with.
    - id: Node Name 
    Edge:
    - relation: relation name
    - file_id: List of text IDs the edge is associated with.
    - type: Type of the edge (e.g., Source, Relation, Concept).
    - synsets: List of synsets associated with the edge.
    
    '''
    g = nx.DiGraph()
    entity_to_id = {}

    # Add triple nodes
    with open(triple_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            g.add_node(get_node_id(node_id, entity_to_id), id=node_id, type=row["type"])
            
    # Add text nodes
    with open(text_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["text_id:ID"]
            g.add_node(node_id, file_id = node_id, id=row["original_text"], type="passage")

    # Add concept nodes
    with open(concept_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["concept_id:ID"]
            g.add_node(node_id, file_id = "concept_file", id=row["name"], type='concept')

    # Add file id for triple nodes and concept nodes when add the edges
    
    # Add triple edges
    with open(triple_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            concepts = ast.literal_eval(row["concepts"])
            g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])
            
            ### ADD CONCEPTS TO THE EDGE ###
            # split the concepts by comma and loop through the list to add synsets
            for concept in concepts:
                if "concepts" not in g.edges[start_id, end_id]:
                    g.edges[start_id, end_id]['concepts'] = str(concept)
                else:
                    g.edges[start_id, end_id]['concepts'] += "," + str(concept)
            

    # Add text edges
    with open(text_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = row[":END_ID"]
            g.add_edge(start_id, end_id, relation='mention in', type=row[":TYPE"])
            
            ### ADD FILE ID TO NODE ###
            if 'file_id' in g.nodes[start_id]:
                g.nodes[start_id]['file_id'] += "," + str(end_id)
            else:
                g.nodes[start_id]['file_id'] = str(end_id)
    # Add concept edges
    with open(concept_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = row[":END_ID"] # end id is concept node id
            g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])

            # if 'file_id' in g.nodes[end_id]:
            #     # split by comma and loop through the list to add file ids
            #     for file_id in g.nodes[start_id]['file_id'].split(','):
            #         g.nodes[end_id]['file_id'] += "," + str(file_id)
            # else:
            #     g.nodes[end_id]['file_id'] = g.nodes[start_id]['file_id']  
    # Write to GraphML
    # check if output file directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nx.write_graphml(g, output_file, infer_numeric_types=True)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert CSV files to GraphML format.')
    parser.add_argument('--triple_node_file', type=str, required=True, help='Path to the triple node CSV file.')
    parser.add_argument('--text_node_file', type=str, required=True, help='Path to the text node CSV file.')
    parser.add_argument('--concept_node_file', type=str, required=True, help='Path to the concept node CSV file.')
    parser.add_argument('--triple_edge_file', type=str, required=True, help='Path to the triple edge CSV file.')
    parser.add_argument('--text_edge_file', type=str, required=True, help='Path to the text edge CSV file.')
    parser.add_argument('--concept_edge_file', type=str, required=True, help='Path to the concept edge CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output GraphML file.')

    args = parser.parse_args()
    
    csvs_to_graphml(args.triple_node_file, args.text_node_file, args.concept_node_file,
                    args.triple_edge_file, args.text_edge_file, args.concept_edge_file,
                    args.output_file)