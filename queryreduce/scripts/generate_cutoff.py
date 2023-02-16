import pandas as pd 
import argparse
import numpy as np
from collections import defaultdict
import pickle 
import os
from math import floor

parser = argparse.ArgumentParser(description='Generate subsets of any array')

parser.add_argument('-source', type=str, help='Dataset Source')
parser.add_argument('-cutoff', type=int, nargs='+')
parser.add_argument('-sink', type=str, help='Output dir')

parser.add_argument('--reverse', action='store_true')


def construct_dict(df):
    out = defaultdict(list)
    for index, row in df.itterows():
        out[row.qid].append((index, row.score))
    return out

def create_record(key, row) -> dict:
    return {'qid': key, 'pid':row[0], 'nid':row[1]}

def get_rows(tuples, cutoff=20, reverse=False) -> list:
    cutoff = cutoff / 100
    triples = []
    for key, item in tuples.items():
        if len(item) < 1: continue
        scored = sorted(item, key=lambda x: x[-1], reverse=reverse)
        count = floor(len(scored) * cutoff)
        if count < 1: cut = [scored[0]]
        else: cut = scored[:count]
        triples.extend([create_record(key, row) for row in cut])
    return triples

def main(args):
    with open(args.source, 'rb') as f:
        df = pickle.load(f)

    for cutoff in args.cutoff:
        triples = pd.DataFrame.from_records(get_rows(df, cutoff, args.reverse))
        name = f'triples.{cutoff}.tsv'

        triples.to_csv(os.path.join(args.sink, name), sep='\t', index=False, header=False)

    return 0

if __name__ == '__main__':
    main(parser.parse_args())