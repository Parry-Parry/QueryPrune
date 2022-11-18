import numpy as np
from queryreduce.distance import init_interpolated_distance 
from queryreduce.config import MarkovConfig
import torch 
from functools import partial

class Process:
    '''
    Markov Process with single ergodic class

    Config Parameters
    -----------------
    alpha : float -> Weight of query embedding on distance 
    beta : float -> Weight of positive document on distance
    triples : np.array -> Set of embeddings of shape [num_samples, num_embeddings, embed_dim]
    distr : torch.distribution -> Some categorical distribution

    Generated Parameters
    --------------------
    P : np.array -> Generate from the number of triples, represents the transition probability matrix 
    distance : func -> Function to calculate a pairwise distance across queries and documents

    run() Parameters
    ----------------
    x0 : int -> id of starting state
    k : int -> desired number of samples
    '''

    state_id = 0
    def __init__(self, config : MarkovConfig) -> None:
        self.P = np.zeros((config.triple.shapes[0], config.triple.shapes[0]), dtype=np.float16)
        self.distance = init_interpolated_distance(config.gpu, config.alpha, config.beta)
        self.triples = config.triples
        self.sample_distr = config.distr

    def _weight(self, x, xs):
        return np.exp(self.distance(x, xs)**2)
    
    def _step(self):
        if np.all(self.P[self.state_id] == 0):
            self.P[self.state_id] = self._weight(np.expand_dims(self.triples[self.state_id], axis=0), self.triples)
        
        distr = self.sample_distr(logits=self.P[self.state_id])
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










