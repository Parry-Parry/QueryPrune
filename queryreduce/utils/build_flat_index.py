import numpy as np 
import faiss 
import logging 
import bz2
import argparse 
import pickle
import multiprocessing as mp

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-k', type=int)
parser.add_argument('-out', type=str) 
parser.add_argument('--compress', action="store_true")

def main(args):
    faiss.omp_set_num_threads(mp.cpu_count())
    if args.compress:
        with bz2.open(args.source, 'rb') as f:
            triples = pickle.load(f)
    else:
        with open(args.source, 'rb') as f:
            triples = np.load(f)

    prob_dim = triples.shape[-1]
    logging.info('Training Index')

    faiss.normalize_L2(triples)
    quantiser = faiss.IndexFlatL2(prob_dim) 
    cpu_index = faiss.IndexIVFFlat(quantiser, prob_dim, args.k, faiss.METRIC_INNER_PRODUCT)
    cpu_index.train(triples)
    cpu_index.add(triples)

    faiss.write_index(cpu_index, args.out + f'triples.{args.k}.index')

    return 0
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Building Faiss IVF Index')
    main(parser.parse_args())