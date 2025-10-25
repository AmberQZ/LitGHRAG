from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder

    @abstractmethod
    def encode(self, query, **kwargs):
        """Abstract method to encode queries."""
        pass
    
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
        if isinstance(self.sentence_encoder, SentenceTransformer):
            if query_prefix:
                query_embeddings = self.sentence_encoder.encode(self.add_eos(query), prompt=query_prefix, normalize_embeddings=normalize_embeddings)
            else:
                query_embeddings = self.sentence_encoder.encode(self.add_eos(query), normalize_embeddings=normalize_embeddings)
        else:
            if query_prefix:
                query_embeddings = self.sentence_encoder.encode(query, instruction=query_prefix, max_length = 32768)
            else:
                query_embeddings = self.sentence_encoder.encode(query, max_length = 32768)

            # Normalize embeddings if required
            if normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1).detach().cpu().numpy()

        # Move to CPU and convert to NumPy
        return query_embeddings

class SentenceEmbedding(BaseEmbeddingModel):
    def __init__(self,sentence_encoder:SentenceTransformer):
        self.sentence_encoder = sentence_encoder

    def encode(self, query, **kwargs):
        normalize_embeddings = kwargs.get('normalize_embeddings', True)
        return self.sentence_encoder.encode(query, normalize_embeddings=normalize_embeddings)
   