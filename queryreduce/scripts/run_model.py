from queryreduce.markov.model import Process
from queryreduce.markov.config import MarkovConfig
import numpy as np 
import argparse
import logging 
import bz2
import pickle
import faiss

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