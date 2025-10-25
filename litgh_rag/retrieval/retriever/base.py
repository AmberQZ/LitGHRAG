from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    """
    Configuration class for inference settings.
    
    Attributes:
        topk (int): Number of top results to retrieve. Default is 5.
        Dmax (int): Maximum depth for search. Default is 4.
        weight_adjust (float): Weight adjustment factor for passage retrieval. Default is 0.05.
        topk_edges (int): Number of top edges to retrieve. Default is 50.
        topk_nodes (int): Number of top nodes to retrieve. Default is 10.
    """
    keyword: str = "musique"
    topk: int = 5
    Dmax: int = 4
    weight_adjust: float = 1.0
    topk_edges: int = 50
    topk_nodes: int = 10
    ppr_alpha: float = 0.99
    ppr_max_iter: int = 2000
    ppr_tol: float = 1e-7


class BaseRetriever(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def retrieve(self, query, topk=5, **kwargs) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("This method should be overridden by subclasses.") 

class BaseEdgeRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def retrieve(self, query, topk=5, **kwargs) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

class BasePassageRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def retrieve(self, query, topk=5, **kwargs) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

