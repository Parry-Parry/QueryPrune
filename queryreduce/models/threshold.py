import time 
import logging

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

from queryreduce.models.config import ThresholdConfig

class Process:
    state_id = 0
    def __init__(self, config : ThresholdConfig) -> None:
        self.triples = config.triples
        self.index = np.arange(len(config.triples)) # Index for candidates
        self.k = config.k # Num samples for mean
        self.t = config.t # Threshold similarity
        self.c = None # Set of Candidates

    def _distance(self, x, mean):
        return np.mean(cosine_similarity(x.reshape(1, -1), mean))

    def _get_indices(self): # Check we have enough candidates to sample
        c = list(self.c)
        l_c = len(c)
        if l_c > self.k:
            return np.random.choice(c, self.k, replace=False)
        else:
            return c 
    
    def _get_candidates(self):
        idx = self._get_indices()

        if len(idx) > 1:
            return self.triples[idx] # Get random K from candidate set
        else:
            return self.triples[idx].reshape(1, -1)

    def _step(self):
        c_id = np.random.choice(self.index) 
        c = self.triples[c_id] # Get random candidate

        K = self._get_candidates()
        d = self._distance(c, K) # Cosine Similarity

        if d < self.t: # If candidate dissimilarity over threshold
            self.state_id = c_id # Accept Candidate

        return self.state_id
    
    def run(self, x0, k):
        self.state_id = x0
        t = 0 
        self.c = set() # Set allows for the compiler to ignore candidates we have already accepted
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        assert self.c is not None
        self.c.add(x0)
        start = time.time()
        while len(self.c) < k:
            self.c.add(self._step())
            t += 1
            if t % 1000: logging.info(f'{t} steps complete, {len(self.c)} candidates found')
        end = time.time() - start 

        logging.info(f'Completed collection in {end} seconds')

        return list(self.c), t