from collections import defaultdict
from typing import Dict
import numpy as np
from queryreduce.distance import init_interpolated_similarity
from queryreduce.config import MarkovConfig
import faiss

class Process:
    '''
    Markov Process with single ergodic class

    Config Parameters
    -----------------
    alpha : float -> Weight of query embedding on distance 
    beta : float -> Weight of positive document on distance
    triples : np.array -> Set of embeddings of shape [num_samples, num_embeddings, embed_dim]
    *Not Used Currently* distr : torch.distribution -> Some categorical distribution

    Generated Parameters
    --------------------
    P : dict[np.array] -> Represents the transition probability matrix 
    distance : func -> Function to calculate a pairwise distance across queries and documents

    run() Parameters
    ----------------
    x0 : int -> id of starting state
    k : int -> desired number of samples
    '''

    state_id = 0
    def __init__(self, config : MarkovConfig) -> None:
        self.triples = config.triples
        self.P : Dict[int, np.array] = defaultdict(lambda : np.zeros(self.triples.shape[0]))
        self.distance = init_interpolated_similarity(config.gpu, config.alpha, config.beta, config.equal)
    
        #self.sample_distr = config.distr

    def _weight(self, x, xs):
        return np.exp(-(self.distance(x, xs)**2))
    
    def _step(self):
        if np.all(self.P[self.state_id] == 0):
            self.P[self.state_id] = self._weight(np.expand_dims(self.triples[self.state_id], axis=0), self.triples)
            faiss.normalize_L2(self.P[self.state_id])
        
        self.state_id = np.random.choice(self.P[self.state_id], 1, p=self.P[self.state_id])

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










