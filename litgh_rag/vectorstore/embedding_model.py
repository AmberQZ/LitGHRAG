from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch.nn.functional as F
from abc import ABC, abstractmethod
import csv

class BaseEmbeddingModel(ABC):
    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder

    @abstractmethod
    def encode(self, query, **kwargs):
        """Abstract method to encode queries."""
        pass
    
    def compute_kg_embedding(self, node_csv_without_emb, node_csv_file, edge_csv_without_emb, edge_csv_file, text_node_csv_without_emb, text_node_csv, **kwargs):
        with open(node_csv_without_emb, "r") as csvfile_node:
            with open(node_csv_file, "w", newline='') as csvfile_node_emb:
                reader_node = csv.reader(csvfile_node)

                # the reader has [name:ID,type,concepts,synsets,:LABEL]
                writer_node = csv.writer(csvfile_node_emb)
                writer_node.writerow(["name:ID", "type", "file_id", "concepts", "synsets", "embedding:STRING", ":LABEL"])

                # the encoding will be processed in batch of 2048
                batch_size = kwargs.get('batch_size', 2048)
                batch_nodes = []
                batch_rows = []
                for row in reader_node:
                    if row[0] == "name:ID":
                        continue
                    batch_nodes.append(row[0])
                    batch_rows.append(row)
                
                    if len(batch_nodes) == batch_size:
                        node_embeddings = self.encode(batch_nodes, batch_size=batch_size, show_progress_bar=False)
                        node_embedding_dict = dict(zip(batch_nodes, node_embeddings))
                        for row in batch_rows:
                        
                            new_row = [row[0], row[1], "", row[2], row[3], node_embedding_dict[row[0]].tolist(), row[4]]
                            writer_node.writerow(new_row)
                            
                        
                        
                        batch_nodes = []
                        batch_rows = []

                if len(batch_nodes) > 0:
                    node_embeddings = self.encode(batch_nodes, batch_size=batch_size, show_progress_bar=False)
                    node_embedding_dict = dict(zip(batch_nodes, node_embeddings))
                    for row in batch_rows:
                        new_row = [row[0], row[1], "", row[2], row[3], node_embedding_dict[row[0]].tolist(), row[4]]
                        writer_node.writerow(new_row)
                    batch_nodes = []
                    batch_rows = []
        

        with open(edge_csv_without_emb, "r") as csvfile_edge:
            with open(edge_csv_file, "w", newline='') as csvfile_edge_emb:
                reader_edge = csv.reader(csvfile_edge)
                # [":START_ID",":END_ID","relation","concepts","synsets",":TYPE"]
                writer_edge = csv.writer(csvfile_edge_emb)
                writer_edge.writerow([":START_ID", ":END_ID", "relation", "file_id", "concepts", "synsets", "embedding:STRING", ":TYPE"])

                # the encoding will be processed in batch of 4096
                batch_size = 2048
                batch_edges = []
                batch_rows = []
                for row in reader_edge:
                    if row[0] == ":START_ID":
                        continue
                    batch_edges.append(" ".join([row[0], row[2], row[1]]))
                    batch_rows.append(row)
                
                    if len(batch_edges) == batch_size:
                        edge_embeddings = self.encode(batch_edges, batch_size=batch_size, show_progress_bar=False)
                        edge_embedding_dict = dict(zip(batch_edges, edge_embeddings))
                        for row in batch_rows:
                            new_row = [row[0], row[1], row[2], "", row[3], row[4], edge_embedding_dict[" ".join([row[0], row[2], row[1]])].tolist(), row[5]]
                            writer_edge.writerow(new_row)
                        batch_edges = []
                        batch_rows = []

                if len(batch_edges) > 0:
                    edge_embeddings = self.encode(batch_edges, batch_size=batch_size, show_progress_bar=False)
                    edge_embedding_dict = dict(zip(batch_edges, edge_embeddings))
                    for row in batch_rows:
                        new_row = [row[0], row[1], row[2], "", row[3], row[4], edge_embedding_dict[" ".join([row[0], row[2], row[1]])].tolist(), row[5]]    
                        writer_edge.writerow(new_row)
                    batch_edges = []
                    batch_rows = []
        

        with open(text_node_csv_without_emb, "r") as csvfile_text_node:
            with open(text_node_csv, "w", newline='') as csvfile_text_node_emb:
                reader_text_node = csv.reader(csvfile_text_node)
                # [text_id:ID,original_text,:LABEL]
                writer_text_node = csv.writer(csvfile_text_node_emb)

                writer_text_node.writerow(["text_id:ID", "original_text", ":LABEL", "embedding:STRING"])

                # the encoding will be processed in batch of 2048
                batch_size = 2048
                batch_text_nodes = []
                batch_rows = []
                for row in reader_text_node:
                    if row[0] == "text_id:ID":
                        continue
                    
                    batch_text_nodes.append(row[1])
                    batch_rows.append(row)
                    if len(batch_text_nodes) == batch_size:
                        text_node_embeddings = self.encode(batch_text_nodes, batch_size=batch_size, show_progress_bar=False)
                        text_node_embedding_dict = dict(zip(batch_text_nodes, text_node_embeddings))
                        for row in batch_rows:
                            embedding  = text_node_embedding_dict[row[1]].tolist()
                            new_row = [row[0], row[1], row[2], embedding]
                            writer_text_node.writerow(new_row)

                        batch_text_nodes = []
                        batch_rows = []

                if len(batch_text_nodes) > 0:
                    text_node_embeddings = self.encode(batch_text_nodes, batch_size=batch_size, show_progress_bar=False)
                    text_node_embedding_dict = dict(zip(batch_text_nodes, text_node_embeddings))
                    for row in batch_rows:
                        embedding  = text_node_embedding_dict[row[1]].tolist()
                        new_row = [row[0], row[1], row[2], embedding]
                        
                        writer_text_node.writerow(new_row)
                    batch_text_nodes = []
                    batch_rows = []

class NvEmbed(BaseEmbeddingModel):
    def __init__(self, sentence_encoder: SentenceTransformer | AutoModel):
        self.sentence_encoder = sentence_encoder

    def add_eos(self, input_examples):
        """Add EOS token to input examples."""
        if self.sentence_encoder.tokenizer.eos_token is not None:
            return [input_example + self.sentence_encoder.tokenizer.eos_token for input_example in input_examples]
        else:
            return input_examples

    def encode(self, query, query_type=None, **kwargs):
        """
        Encode the query into embeddings.
        
        Args:
            query: Input text or list of texts.
            query_type: Type of query (e.g., 'passage', 'entity', 'edge', 'fill_in_edge', 'search').
            **kwargs: Additional arguments (e.g., normalize_embeddings).
        
        Returns:
            Embeddings as a NumPy array.
        """
        normalize_embeddings = kwargs.get('normalize_embeddings', True)

        # Define prompt prefixes based on query type
        prompt_prefixes = {
            'passage': 'Given a question, retrieve relevant documents that best answer the question.',
            'entity': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
            'edge': 'Given a question, retrieve relevant triplet facts that matches this question.',
            'fill_in_edge': 'Given a triples with only head and relation, retrieve relevant triplet facts that best fill the atomic query.'
        }

        if query_type in prompt_prefixes:
            prompt_prefix = prompt_prefixes[query_type]
            query_prefix = f"Instruct: {prompt_prefix}\nQuery: "
        else:
            query_prefix = None

        # Encode the query
        if issubclass(type(self.sentence_encoder), SentenceTransformer):
            if query_prefix:
                query_embeddings = self.sentence_encoder.encode(self.add_eos(query), prompt=query_prefix, **kwargs)
            else:
                query_embeddings = self.sentence_encoder.encode(self.add_eos(query), **kwargs)
        elif issubclass(type(self.sentence_encoder), AutoModel):
            if query_prefix:
                query_embeddings = self.sentence_encoder.encode(query, instruction=query_prefix, max_length = 32768, **kwargs)
            else:
                query_embeddings = self.sentence_encoder.encode(query, max_length = 32768, **kwargs)

            # Normalize embeddings if required
            if normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1).detach().cpu().numpy()

        # Move to CPU and convert to NumPy
        return query_embeddings

class SentenceEmbedding(BaseEmbeddingModel):
    def __init__(self,sentence_encoder:SentenceTransformer):
        self.sentence_encoder = sentence_encoder

    def encode(self, query, **kwargs):
        return self.sentence_encoder.encode(query, **kwargs)
