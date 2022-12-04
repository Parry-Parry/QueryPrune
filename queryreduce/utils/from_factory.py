import numpy as np 
import faiss 
import logging 
import bz2
import argparse 
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-target_dim', type=int)
parser.add_argument('-factory', type=str)
parser.add_argument('-code', type=int)
parser.add_argument('-out', type=str) 
parser.add_argument('--compress', action="store_true")  
parser.add_argument('--l2', action="store_true")  

def main(args):
    if args.compress:
        with bz2.open(args.source, 'rb') as f:
            triples = pickle.load(f)
    else:
        with open(args.source, 'rb') as f:
            triples = np.load(f)

    prob_dim = triples.shape[-1]
    logging.info('Training Index')

    if not args.l2: metric = faiss.METRIC_INNER_PRODUCT
    else: metric = faiss.METRIC_L2

    faiss.normalize_L2(triples)
    index = faiss.index_factory(args.code, args.factory, metric)
    index.train(triples)
    index.add(triples)

    suffix = args.factory.strip(',')

    faiss.write_index(index, args.out + f'triples.{args.code}.{suffix}.index')

    return 0
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Building Faiss IVF Index')
    main(parser.parse_args())