import faiss 
import numpy as np
import logging 
from queryreduce.models.config import MetropolisConfig

class Sampler:
    
    def __init__(self, config : MetropolisConfig) -> None:
        compare = {
            'max' : self._compare_max,
            'min' : self._compare_mean
        }
        distr_ll = {
            'gaussian' : self._gaussian_ll
        }

        self.states = config.states
        self.alpha = config.alpha
        self.warmup = config.warmup

        self.mu = np.mean(config.states, axis=1)
        self.sigma_curr = config.sigma if config.sigma else 1.
        self.sigma_prev = self.sigma_curr

        self.ll = distr_ll[config.distr]

        pass

    

    def _gaussian_transition(self):
        return 

    def _gaussian_ll(self, x):
        pass

    def sample(self) -> np.array:
        pass 

    def _get_candidate(self) -> np.array:
        pass 

    def run(self, x0, k) -> np.array:



