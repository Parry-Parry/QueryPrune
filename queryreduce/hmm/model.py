import numpy as np
from queryreduce.distance.distance import init_distance 
import torch 
from functools import partial

class Process:
    '''
    Markov Process with single ergodic class

    Config Parameters
    -----------------
    n_states : int -> Number of samples that can be reached
    alpha : float -> Weight of query embedding on distance 
    beta : float -> Weight of positive document on distance
    index : faiss.index -> Indexed samples for distance search
    triples : List[NamedTuple] -> Set of embeddings
    distr : torch.distribution -> Some categorical distribution

    run() Parameters
    ----------------
    x0 : int -> id of starting state
    k : int -> desired number of samples
    '''

    state_id = 0
    def __init__(self, config) -> None:
        self.P = np.zeros((config.triple.shape[0], config.triple.shape[0]), dtype=np.float16)
        self.distance = init_distance(config.gpu, config.alpha, config.beta)
        self.triples = config.triples
        self.sample_distr = config.distr

    def _distance(self, x, xs):
        return np.exp(self.distance(x, xs)**2)
    
    def _step(self):
        if np.all(self.P[self.state_id] == 0):
            triple = self.triples[self.state_idx]
            self.P[self.state_idx] = self._distance(triple, self.triples)
        
        distr = self.sample_distr(logits=self.P[self.state_idx])
        self.state_id = distr.sample()

        return self.state_id
    
    def run(self, x0, k):
        self.state_id = x0
        t = 0 
        idx = []
        while len(idx) < k:
            candidate = self._step()
            if candidate not in idx: idx.append(candidate)
            t += 1
        
        return np.array(idx), t










