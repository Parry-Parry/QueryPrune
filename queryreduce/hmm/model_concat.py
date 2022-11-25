from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
from queryreduce.config import MarkovConfig
from queryreduce.utils.utils import weight
import faiss
import logging
import time

class Process:
    '''
    Markov Process with single ergodic class

    Config Parameters
    -----------------
    alpha : float -> Weight of query embedding on distance 
    beta : float -> Weight of positive document on distance
    equal : bool -> If True apply no weighting to embedding space
    dim : int -> dimensionality of single embedding
    k : int -> number of clusters in Index
    n : int -> n-nearest neighbours in similarity search 
    triples : np.array -> Set of embeddings of shape [num_samples, num_embeddings * embed_dim]

    Generated Parameters
    --------------------
    P : dict[np.array] -> Represents the transition probability matrix 
    prob_dim : int -> Derived from 3 * embed dim
    index : faiss.index -> Cosine Similarity Search

    run() Parameters
    ----------------
    x0 : int -> id of starting state
    k : int -> desired number of samples
    '''

    state_id = 0
    def __init__(self, config : MarkovConfig) -> None:
        self.triples = weight(config.triples, config.dim, config.alpha, config.beta, config.equal)
        self.P : Dict[int, Tuple[np.array, np.array]] = defaultdict(lambda : (np.zeros(config.n), np.zeros(config.n)))
        self.prob_dim = 3 * config.dim
        self.index = self._build_index(self.triples, config.k)
        self.n = config.n
    
    def _build_index(self, triples : np.array, k : int):
        ngpus = faiss.get_num_gpus()
        if ngpus < 1:
            logging.error("Error! Faiss Indexing Requires GPU, Exiting...")
            exit

        logging.info('Building Index')
        faiss.normalize_L2(triples)
        quantiser = faiss.IndexFlatL2(self.prob_dim) 
        cpu_index = faiss.IndexIVFFlat(quantiser, self.prob_dim, k, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(triples)

        if ngpus > 1:
            index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            res = faiss.StandardGpuResources() 
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        index.add(triples)
        return index

    def _distance(self, x):
        return self.index.search(-x, self.n)

    def _weight(self, x):
        D, I = self._distance(x)
        gaussian_distance = np.exp(-np.square(D))
        probs = gaussian_distance / np.linalg.norm(gaussian_distance)

        return probs, I
    
    def _step(self):
        if np.all(self.P[self.state_id][0] == 0):
            probs, I = self._weight(np.expand_dims(self.triples[self.state_id], axis=0))
            self.P[self.state_id] = (probs, I)
        
        probs, I = self.P[self.state_id]    
        self.state_id = np.random.choice(I, 1, p=probs)

        return self.state_id
    
    def run(self, x0, k):
        self.state_id = x0
        t = 0 
        idx = set()
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        while len(idx) < k:
            candidate = self._step()
            if candidate not in idx: idx.add(candidate)
            t += 1
        end = time.time() 

        seconds = end - start 
        minutes = seconds / 60
        hours = minutes / 60 

        if hours > 1:
            logging.info(f'Completed search in {hours} hours with {t} steps')
        elif minutes > 1:
            logging.info(f'Completed search in {minutes} minutes with {t} steps')
        else:
            logging.info(f'Completed search in {seconds} seconds with {t} steps')

        return np.array(idx), t