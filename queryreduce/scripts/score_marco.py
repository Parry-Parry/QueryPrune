from collections import defaultdict
import logging
import os
import pickle
import time
import pyterrier as pt
pt.init()

from pyterrier_pisa import PisaIndex
import ir_datasets
import pandas as pd
import numpy as np
import argparse

def make_score(df):
    tmp = defaultdict(dict)
    for row in df.itertuples():
        tmp[row.qid][row.docno] = row.score
    return tmp

def make_triples(df):
    tmp = defaultdict(list)
    for row in df.itertuples():
        tmp[row.qid].append((row.pid, row.nid))
    return tmp

def score_triples(results, triples):

    def apply_score(pos_id, neg_id):
        try:
            pos = results[pos_id]
        except KeyError:
            return (pos_id, neg_id, 0)
        try:
            neg = results[neg_id]
        except KeyError:
            return (pos_id, neg_id, 1e4)
        return (pos_id, neg_id, abs(pos - neg))

    scores = [apply_score(row[0], row[1]) for row in triples]

    return scores
    
parser = argparse.ArgumentParser(description='Generate subsets of any array')

parser.add_argument('-out', type=str, help='Output dir for tsv')
parser.add_argument('-threads', type=int, help='Num CPU Cores')

parser.add_argument('--verbose', action='store_true')

def main(args):
    ds = ir_datasets.load("msmarco-passage/train/triples-small")
    triples = pd.DataFrame(ds.docpairs_iter()).rename(columns={'query_id':'qid', 'doc_id_a':'pid', 'doc_id_b':'nid'})
    queries = pd.DataFrame(ds.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    pisa_index = PisaIndex.from_dataset('msmarco_passage', 'pisa_porter2')
    logging.info(f'Creating index with {args.threads} threads')
    index = pisa_index.bm25(num_results=1000, threads=args.threads)

    start = time.time()
    scores = index.transform(queries)
    end = time.time() - start

    logging.info(f'Queries mined in {end} seconds')
    logging.info('Grouping Scores & Triples')

    del index
    del queries 
    start = time.time()

    score_queries = make_score(scores)
    del scores
    triple_queries = make_triples(triples)
    del triples 

    end = time.time() - start
    logging.info(f'Groups created in {end} seconds')

    counter = 0 
    frames = {}

    start = time.time()
        
    for q in ds.queries_iter():
        qid = q.query_id
        _triples = triple_queries[qid]
        _scores = score_queries[qid]
        frames[qid] = score_triples(_scores, _triples)
        counter += 1
        logging.debug(f'Current Counter: {counter}')
        if counter % 10 == 0:
            end = time.time() - start
            logging.info(f'Last 10 queries processed in {end} seconds, {end / 10} p/query')
            start = time.time()

    with open(os.path.join(args.out, 'scored.pkl'), 'wb') as f:
        pickle.dump(frames, f)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    main(args)