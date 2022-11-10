from typing import Any, Union
from dataclasses import dataclass
from queryreduce.utils.config import EmbedConfig
from torch.nn import tensor
import numpy as np

@dataclass
class Triplet:
    q : Any
    d_pos : Any 
    d_neg : Any

class EmbeddingWrapper():
    def __init__(self, config : EmbedConfig, **kwargs) -> None:
        self.index = config.index
        self.doc_LM = config.doc_LM

        if config.sep_query_LM:
            self.query_LM = config.query_LM
        else:
            self.query_LM = config.doc_LM
    
    def _retrieve(id : Union[int, str], is_q):
        return None if not is_q else None

    def _embed(self, txt : str, is_q=False) -> tensor:
        return self.doc_LM(txt) if not is_q else self.query_LM(txt)

    def create_triplet(self, qid : Union[int, str], dposid : Union[int, str], dnegid : Union[int, str]) -> Triplet:
        retr = lambda x, y : self._retrieve(x, y)

        q = self._embed(retr(qid, True), True)
        d_pos = self._embed(retr(dposid, False))
        d_neg = self._embed(retr(dnegid, False))

        return Triplet(q, d_pos, d_neg)

    def create_representation(self, triplet : Triplet) -> tensor:
        return np.mean([triplet.q, triplet.d_pos, triplet.d_neg], axis=1)