from typing import Any, Union, NamedTuple
from queryreduce.config import EmbedConfig, Embedding, Triplet
from torch.nn import tensor
import numpy as np

'''
Retrieve text associated with triplets of IDs, generate embeddings and return tuples
'''

class EmbeddingWrapper():
    def __init__(self, config : EmbedConfig, **kwargs) -> None:
        self.model = config.model
        self.tokenizer = config.tokenizer

        self.dataset = config.dataset 
        self.store = config.dataset.doc_store()
        self.queries = [query for query in config.dataset.queries_iter()]
        self.q_ids = [query.id for query in self.queries]
    
    def _retrieve(self, id : Union[int, str], is_q):
        return self.store.get(id) if not is_q else self.queries[self.q_ids.index(id)]

    def _embed(self, id : Union[int, str], is_q : bool =False) -> tensor:
        txt = self._retrieve(id, is_q)
        tok, mask = self.tokenizer(txt) 
        embedding = self.model.doc(tok, mask) if not is_q else self.model.query(tok, mask)
        return embedding

    def create_triplet(self, qid : Union[int, str], dposid : Union[int, str], dnegid : Union[int, str]) -> Triplet:
        q = self._embed(qid, True)
        d_pos = self._embed(dposid, False)
        d_neg = self._embed(dnegid, False)

        return Triplet(qid, q, d_pos, d_neg)
    
    def get_docs(self, ids : list) -> list:
        return self.store.get_many(ids)
    
    def get_queries(self, qids : list) -> list:
        return [self.queries[self.q_ids.index(id)] for id in qids]
    
