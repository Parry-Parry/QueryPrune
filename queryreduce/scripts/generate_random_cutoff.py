import logging
import pandas as pd 
import argparse
import numpy as np
from collections import defaultdict
import pickle 
import os
from math import floor
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate subsets of any array')

parser.add_argument('-source', type=str, help='Dataset Source')
parser.add_argument('-cutoff', type=int, nargs='+')
parser.add_argument('-sink', type=str, help='Output dir')

parser.add_argument('--verbose', action='store_true')

def construct_tuple(row):
    return (row.query, row.score)

def construct_dict(df):
    out = defaultdict(list)
    for index, row in df.items():
        out[row[0]].append((index, row[1]))
    return out

def get_random(tuples, cutoff=20):
    cutoff = cutoff / 100
    idx = []
    for key, item in tuples.items():
        count = max(1, floor(len(item) * cutoff))
        logging.debug(f'For query {key}, choose {count} items')
        tmp_idx = np.random.choice(item, count, replace=False)
        idx.extend(tmp_idx)
    return np.array(idx)

def open_bm25(file):
    with open(file, 'rb') as f:
        idx, score = pickle.load(f)
    return idx, score

def main(args):
    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
   
    df = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    

    idx = defaultdict(list)
    logging.info('Building Lookup')

    for index, row in df.iterrows():
        idx[row.query].append(index)
    

    for cutoff in args.cutoff:
        logging.info(f'Collecting top {cutoff}%')
        out_idx = get_random(idx, cutoff)
        logging.info(f'For {cutoff}%, {len(out_idx)} mined')
        name = f'random.{len(df)}.{cutoff}.pkl'

        with open(os.path.join(args.sink, name), 'wb') as f:
            pickle.dump((out_idx, None), f)
    
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('Collecting Query Ordered Random Subset')
    main(args)