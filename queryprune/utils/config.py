from dataclasses import dataclass
from typing import NamedTuple, Any, Union
import torch
from numpy import array

@dataclass
class Text:
    id : Union[int, str]
    embedding : torch.tensor

    def distance(self, other):
        return torch.linalg.norm(other.embedding - self.embedding)

@dataclass
class TextSet:
    ids: list 
    embeddings : array 

class EmbedConfig(NamedTuple):
    sep_query_LM : bool 
    doc_LM : Any 
    query_LM : Any

