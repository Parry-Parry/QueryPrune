import faiss 
import numpy as np
import logging 
from functools import partial
from queryreduce.models.config import AcceptanceConfig

'''
Rejection Sampler Based on a Rolling Centroid

Parameters
----------

states : np.array -> Vectors representing discrete state soace
metric : str -> What type of metric to measure distance by
sub : int -> Subset of previous candidates for each comparison 
alpha : float -> Acceptance threshold for ratio
update : int -> How often to recompute centroid 
compare : str -> Take either maximum or mean for ratio
gpus : int -> Number of usable gpus

Generated Parameters
--------------------

id : np.array -> Seperate index array to allow removal of candidates without having to run through high dim state space
centroid : np.array -> Rolling centroid of candidate cluster for acceptance ratio

Runtime Parameters
------------------

x0 : int -> Starting index
k : int -> Desired number of samples
'''

class Sampler:
    def __init__(self, config : AcceptanceConfig) -> None:
        compare = {
            'max' : self._compare_max,
            'min' : self._compare_mean
        }
        distance = {
            'L2' : faiss.METRIC_L2,
            'IP' : faiss.METRIC_INNER_PRODUCT
        }

        self.states = config.states 
        if config.metric == 'IP' : faiss.normalize_L2(self.states)
        self.metric = distance[config.metric]
        self.id = np.arange(len(config.states), dtype=np.int64)
        self.sub = config.sub
        self.alpha = config.alpha
        self.update = config.update

        self.idx = set()
        self.centroid = np.zeros(config.states.shape[-1])
        self.compare = compare[config.compare]

        if config.gpus:
            res = [faiss.StandardGpuResources() for _ in config.gpus]
            self.distance = partial(faiss.pairwise_distance_gpu, res=res)
        else:
            self.distance = faiss.pairwise_distances
    
    def _compare_max(self, x, c) -> float:
        return np.max(x) / np.max(c)

    def _compare_mean(self, x, c) -> float:
        return np.mean(x) / np.mean(c)

    def _threshold(self, x):
        '''
        Case:
            * There are more candidates than the desired subset
            * There are no candidates
            * There are less candidates than the desired subset
        '''
        if len(self.idx) > self.sub:
            indices = np.random.choice(self.idx, self.sub)
        elif len(self.idx) == 0:
            return 10.
        else:
            indices = self.idx
        
        vecs = self.states[indices]
        dist_x = self.distance(x, vecs, metric=self.metric)
        dist_c = self.distance(self.centroid, vecs, metric=self.metric)

        return self.compare(dist_x, dist_c)

    def run(self, x0, k) -> np.array:
        x_init = self.states[x0]
        self.centroid = x_init
        ticker = 0 # Update Ticker
        t = 0 # Total Steps
        while len(self.idx) < k:
            x_cand = np.random.choice(self.id)
            np.delete(self.id, x_cand)
            threshold = self._threshold(self.states[x_cand])
            if threshold > self.alpha:
                self.idx.add(x_cand)
                if ticker % self.update == 0:
                    self.centroid = np.mean(self.states[self.idx])
                ticker += 1
            t += 1
        
        return np.array(list(self.idx)), t
            
            
            









