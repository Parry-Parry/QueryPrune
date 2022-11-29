import numpy as np 
import argparse
import logging 
import bz2
import pickle
import faiss
import time 
from typing import Dict, Tuple, Any, NamedTuple
from collections import defaultdict

def weight(array : np.array, dim : int, alpha : float, beta : float, equal : bool) -> np.array:
    
    if equal: 
        faiss.normalize_L2(array)
        return array

    gamma = np.max(1 - alpha - beta, 0)

    array[:, :dim] = alpha * array[:, :dim]
    array[:, dim:2*dim] = beta * array[:, dim:2*dim]
    array[:, 2*dim:3*dim] = gamma * array[:, 2*dim:3*dim]
    faiss.normalize_L2(array)
    return array

class MarkovConfig(NamedTuple):
    triples : Any
    dim : int
    alpha : float
    beta : float
    equal : bool
    n : int
    k : int 
    store : str 
    nprobe : int
    built : bool


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
    triples : np.array -> Set of embeddings of shape [num_samples, num_embeddings * embed_dim] **NORMALISED**

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
        self.nprobe = config.nprobe
        self.index = self._build_index(self.triples, config.k, config.store) if not config.built else self._load_index(config.store, config.k)
        self.n = config.n
    
    def set_nprobe(self, nprobe):
        self.nprobe = nprobe 
        self.index.nprobe = nprobe
    
    def _load_index(self, store : str, k : int):
        assert store is not None
        ngpus = faiss.get_num_gpus()
        if ngpus < 1:
            logging.error("Error! Faiss Indexing Requires GPU, Exiting...")
            exit

        cpu_index = faiss.read_index(store + f'triples.{k}.index')
        if ngpus > 1:
            index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            res = faiss.StandardGpuResources() 
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        index.nprobe = self.nprobe
            
        return index
    
    def _build_index(self, triples : np.array, k : int, store : str):
        assert store is not None
        ngpus = faiss.get_num_gpus()
        if ngpus < 1:
            logging.error("Error! Faiss Indexing Requires GPU, Exiting...")
            exit

        logging.info('Building Index...')

        quantiser = faiss.IndexFlatL2(self.prob_dim) 
        cpu_index = faiss.IndexIVFFlat(quantiser, self.prob_dim, k, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(triples)
        cpu_index.add(triples)

        logging.info('Storing Index to Disk...')
        faiss.write_index(cpu_index, store + f'triples.{k}.index')

        if ngpus > 1:
            index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            res = faiss.StandardGpuResources() 
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        index.nprobe = self.nprobe

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
            idx.add(self._step())
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

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-alpha', type=float, default=1.)
parser.add_argument('-beta', type=float, default=1.)
parser.add_argument('-n', type=int, default=1)
parser.add_argument('-k', type=int, default=100)
parser.add_argument('-store', type=str)
parser.add_argument('-samples', type=int, default=1)
parser.add_argument('-out', type=str, default='/')
parser.add_argument('-nprobe', type=int, default=0)
parser.add_argument('--eq')
parser.add_argument('--built')

def main(args):
    with bz2.open(args.source, 'rb') as f:
        array = pickle.load(f)
    
    nprobe = args.k // 10 if args.nprobe == 0 else args.nprobe
   
    config = MarkovConfig(
        triples=array,
        dim=array.shape[-1]//3,
        alpha=args.alpha,
        beta=args.beta,
        equal=True if args.eq else False,
        n = args.n,
        k = args.k,
        store = args.store,
        nprobe = nprobe, 
        built = True if args.built else False
    )

    model = Process(config)
    I, t = model.run(np.random.randint(len(config.triples)), args.samples)

    logging.info(f'{args.samples} samples found in {t} steps, Saving...')

    with open(args.out + f'samples.{args.samples}.pkl', 'wb') as f:
        pickle.dump(I, f)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('--Initialising Candidate Choice Using Markov Process--')
    main(parser.parse_args())