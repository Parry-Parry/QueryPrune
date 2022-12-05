from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
from queryreduce.models.config import MarkovConfig
from queryreduce.utils.utils import time_output, to_device, weight
import faiss
import logging
import time
import multiprocessing as mp

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
    def __init__(self, config : MarkovConfig) -> None:
        self.triples = weight(config.triples, config.dim, config.alpha, config.beta, config.equal)
        self.P : Dict[int, np.array, np.array] = defaultdict(lambda : np.zeros(config.n))
        self.batch = config.batch
        self.prob_dim = 3 * config.dim
        self.nprobe = config.nprobe
        self.ngpu = config.ngpu
        self.index = self._load_index(config.store) if config.built else self._build_index(config.triples, config.k, config.store)
        self.n = config.n
        self.state_idx = np.zeros(config.batch)
        if self.ngpu > 0:
            logging.info('Using GPU, capping neighbours at 2048')
            self.n = min(2048, self.n)

    
    def set_nprobe(self, nprobe : int):
        self.nprobe = nprobe 
        self.index.nprobe = nprobe
    
    def _load_index(self, store : str):
        assert store != ''

        index = faiss.read_index(store)
        index.nprobe = self.nprobe
        gpu = to_device(index. self.ngpu)

        return index

    def _distance(self, x):
        return self.index.search(-x, self.n)
    
    def _get_batch(self, id : int) -> None:
        tmp_id = id 
        self.state_idx[0] = id
        for i in range(1, self.batch):
            _, I = self._distance(np.expand_dims(self.triples[tmp_id], axis=0))
            self.state_idx[i] = np.random.choice(I.ravel())
        self.state_idx = self.state_idx.astype(np.int64)
        logging.info('First Batch Found, Starting Search...')
        
    def _step(self) -> np.array:
        _, I = self._distance(self.triples[self.state_idx])
        self.state_idx = np.apply_along_axis(np.random.choice, 1, np.reshape(I, (self.batch, self.n)))

        return self.state_idx
    
    def run(self, x0, k):
        faiss.omp_set_num_threads(mp.cpu_count())
        t = 0 
        idx = set()
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        accum = 0
        self._get_batch(x0)
        while len(idx) < k:
            batch_time = time.time()
            idx.update(list(self._step()))
            diff = time.time() - batch_time
            accum += diff
            if t % 100==0: 
                logging.info(f'Last 100 steps complete in {accum} seconds | {accum / (self.batch * 100)}  seconds p/batch | {len(idx)} candidates found')
                accum = 0
            t += 1
        end = time.time()
        logging.info(time_output(end - start))

        return np.array(list(idx))[:k], t