import numpy as np 
import faiss 
import logging 
import bz2
import argparse 
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-k', type=int)
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


    faiss.normalize_L2(triples)
    quantiser = faiss.IndexHNSWFlat(prob_dim, 64) 
    if not args.l2: index = index = faiss.IndexIVFFlat(quantiser, prob_dim, args.k)
    else: faiss.IndexIVFFlat(quantiser, prob_dim, args.k, faiss.METRIC_L2)

    index.cp.min_points_per_centroid = 5
    index.quantizer_trains_alone = 2

    index.train(triples)
    index.add(triples)

    logging.info('Trained Index, Saving...')
    faiss.write_index(index, args.out + f'hnsw.{args.k}.index')

    return 0
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Building Faiss IVF Index')
    main(parser.parse_args())