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
    triples : np.array -> Set of embeddings of shape [num_samples, num_embeddings * embed_dim] 
    dim : int -> dimensionality of single embedding
    alpha : float -> Weight of query embedding on distance 
    beta : float -> Weight of positive document on distance
    equal : bool -> If True apply no weighting to embedding space
    batch : int -> Batch size of query search
    batch_type : str -> mean or std batch intialization
    n : int -> n-nearest neighbours in similarity search 
    store : str -> Path to index
    nprobe : int -> How many index cells to check
    ngpu : int -> Number of GPUs

    Generated Parameters
    --------------------
    triples : np.array -> Weighted embedding matrix
    prob_dim : int -> Derived from 3 * embed dim
    state_idx : np.array -> Current Batch Indices

    run() Parameters
    ----------------
    x0 : int -> id of starting state
    k : int -> desired number of samples
    '''
    def __init__(self, config : MarkovConfig) -> None:

        batching = {
            'std' : self._get_batch,
            'mean' : self._get_mean_batch
        }

        self.triples = weight(config.triples, config.dim, config.alpha, config.beta, config.equal)
        self.P : Dict[int, np.array, np.array] = defaultdict(lambda : np.zeros(config.n))
        self.batch = config.batch
        self.prob_dim = 3 * config.dim
        self.nprobe = config.nprobe
        self.ngpu = config.ngpu
        self.index = self._load_index(config.store) 
        self.n = config.n
        self.state_idx = np.zeros(config.batch, dtype=np.int64)
        self.cache = defaultdict(lambda x : np.zeros(config.n, dtype=np.int64))
        if self.ngpu > 0:
            logging.info('Using GPU, capping neighbours at 2048')
            self.n = min(2048, self.n)
        self.get_batch = batching[config.batch_type]
    
    def _load_index(self, store : str):
        assert store != ''

        index = faiss.read_index(store)
        index.nprobe = self.nprobe
        gpu = to_device(index, self.ngpu)

        return index
    
    def _expand(self, x):
        return np.expand_dims(x, axis=0)

    def _distance(self, x : np.array) -> np.array:
        _, I = self.index.search(-x, self.n)
        return I.ravel()
    
    def _expand_distance(self, x : np.array) -> np.array:
        return self._distance(self._expand(x))
    
    def _get_mean_batch(self, id: int) -> None:
        self.state_idx[0] = id
        self.state_idx[1] = int(np.random.choice(self._expand_distance(self.triples[self.state_idx[0]])))
        for i in range(2, self.batch):
            vec = np.mean([self._expand(self.triples[self.state_idx[i-1]]), self._expand(self.triples[self.state_idx[i-2]])], axis=0)
            candidates = self._distance(vec)
            self.cache[self.state_idx[i-1]] = candidates
            self.state_idx[i] = np.random.choice(candidates)

        self.state_idx = self.state_idx.astype(np.int64)

    def _get_batch(self, id : int) -> None:
        self.state_idx[0] = id
        for i in range(1, self.batch):
            candidates = self._expand_distance(self.triples[self.state_idx[i-1]])
            self.cache[self.state_idx[i-1]] = candidates
            self.state_idx[i] = np.random.choice(candidates)
        self.state_idx = self.state_idx.astype(np.int64)

    def _retrieve(self, x : np.array) -> np.array:
        result = np.stack([self.cache[id] for id in x])
        return result
    
    def _choice(self, x):
        vec_in = np.vectorize(lambda x : x in self.cache)
        
        
    def _step(self) -> np.array:
        vec_in = np.vectorize(lambda x : x in self.cache)
        filter = vec_in(self.state_idx)

        if len(filter != 0) and len(filter != self.batch):
            tmp_array = np.zeros((self.batch, self.n), dtype=np.int64)
            cached = self.state_idx[filter]
            compute = self.state_idx[np.logical_not(filter)]
            tmp_array[filter] = self._retrieve(cached)
            computed = np.reshape(self._distance(self.triples[compute]), (len(compute), self.n))
            tmp_array[np.logical_not(filter)] = computed
            for key, value in zip(compute, computed):
                self.cache[key] = value

            self.state_idx = np.apply_along_axis(np.random.choice, 1, tmp_array)
        elif len(filter==self.batch):
            self.state_idx = np.apply_along_axis(np.random.choice, 1, self._retrieve(self.state_idx))
        else:
            self.state_idx = np.apply_along_axis(np.random.choice, 1, np.reshape(self._distance(self.triples[self.state_idx]), (self.batch, self.n)))
            
        return self.state_idx
    
    def run(self, x0 : int, k : int) -> Tuple[np.array, int]:
        faiss.omp_set_num_threads(mp.cpu_count())
        t = 0 
        idx = set()
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        accum = 0
        self.get_batch(x0)
        logging.info('First Batch Found, Starting Search...')
        while len(idx) < k:
            batch_time = time.time()
            idx.update(list(self._step()))
            diff = time.time() - batch_time
            accum += diff
            if t % 100==0: 
                logging.info(f'Last 100 steps complete in {accum} seconds | {accum / (self.batch * 100)}  seconds per query | {len(idx)} candidates found')
                accum = 0
            t += 1
        end = time.time()
        logging.info(time_output(end - start))

        return np.array(list(idx))[:k], t