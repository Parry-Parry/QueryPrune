import time
import faiss 
import numpy as np
import logging 
import multiprocessing as mp

from queryreduce.models.config import CentroidConfig
from queryreduce.utils.utils import time_output

'''
Rejection Sampler Based on a Rolling Centroid

Parameters
----------

states : np.array -> Vectors representing discrete state soace
sub : int -> Subset of previous candidates for each comparison
update : int -> How often to recompute centroid 
compare : str -> Take either maximum or mean for ratio
threshold : str -> Use a random or precomputed subset comparison
threshold_init : float -> Threshold value for similarity with initial centroid
gpus : int -> Number of usable gpus

Generated Parameters
--------------------

id : np.array -> Seperate index array to allow removal of candidates without having to run through high dim state space
centroid : np.array -> Rolling centroid of candidate cluster for acceptance ratio
subset : np.array -> Storage of subset used as candidate cluser

Runtime Parameters
------------------

x0 : int -> Starting index
k : int -> Desired number of samples
'''

class Sampler:
    def __init__(self, config : CentroidConfig) -> None:
        faiss.omp_set_num_threads(mp.cpu_count())
        compare = {
            'max' : self._compare_max,
            'mean' : self._compare_mean
        }
        threshold = {
            'random' : self._random_threshold,
            'set' : self._set_threshold
        }

        self.states = config.states
        faiss.normalize_L2(self.states)
        self.id = np.arange(len(config.states), dtype=np.int64)
        self.sub = config.sub
        self.update = config.update

        self.idx = []
        self.centroid = np.zeros((1, config.states.shape[-1]))
        self.subset = None
        self.threshold_val = config.threshold_init
        self.compare = compare[config.compare]
        self.threshold = threshold[config.threshold]
        
        self.distance = np.inner
    
    def _compare_max(self, x, xs) -> float:
        return np.max(np.inner(x, xs)) < self.threshold_val

    def _compare_mean(self, x, xs) -> float:
        return np.mean(np.inner(x, xs)) < self.threshold_val

    def _get_subset(self) -> np.array:
        if len(self.idx) > self.sub:
            logging.debug('More candidates than subset')
            indices = np.random.choice(self.idx, self.sub, replace=False)
        elif len(self.idx) == 0:
            logging.debug('No Candidates')
            return None
        else:
            logging.debug('Less candidates than subset')
            indices = self.idx
        return self.states[indices]

    def _update_centroid(self) -> None:
        candidates = self._get_subset()
        if candidates is None: return None
        self.subset = candidates
        self.centroid = np.expand_dims(np.mean(candidates, axis=0), axis=0)
        self.threshold_val = np.mean(self.distance(self.centroid, candidates))
    
    def _set_threshold(self, x):
        if self.subset is not None: return self.compare(x, self.subset)
        return self.compare(x, self.centroid)

    def _random_threshold(self, x):
        vecs = self._get_subset()

        if not vecs: return True
        return self.compare(x, vecs)

    def run(self, x0, k) -> np.array:
        x_init = self.states[x0]
        self.centroid = np.expand_dims(x_init, axis=0)
        ticker = 0 # Update Ticker
        t = 0 # Total Steps
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        while len(self.idx) < k:
            x_cand = np.random.choice(self.id)
            self.id = self.id[np.where(self.id != x_cand)]
            threshold = self.threshold(np.expand_dims(self.states[x_cand], axis=0))
            logging.debug(f'Threshold value {threshold}')
            if threshold:
                ticker += 1
                self.idx.append(x_cand)
            
            if t % self.update == 0:
                logging.debug(f'Updating Centroid at step {t}')
                self._update_centroid()
                
            if t % 1000 == 0:
                diff = time.time() - start
                logging.info(f'Time Elapsed over {t} steps: {diff} | {len(self.idx)} candidates found')
            t += 1
        end = time.time()
        logging.info(time_output(end - start))
        
        return np.array(list(self.idx)), t