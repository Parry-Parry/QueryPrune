import logging
import os
import pyterrier as pt
pt.init()

from pyterrier_pisa import PisaIndex
import ir_datasets
import pandas as pd
import argparse

class Mine:
    def __init__(self, ds) -> None:
        queries = pd.DataFrame(ds.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'}).set_index('qid')
        self.queries = queries['query'].to_dict()
        self.documents = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    
    def get_batch(self, triples) -> pd.DataFrame:
        _df = {'query':[], 'psg+':[], 'psg-':[]}
        for item in triples.itertuples():
            _df['query'].append(self.queries[item.q])
            _df['psg+'].append(self.documents[item.pos])
            _df['psg-'].append(self.documents[item.neg])

        return pd.DataFrame(_df)

parser = argparse.ArgumentParser(description='Generate subsets of any array')

parser.add_argument('-ids', type=str, help='Source IDs')
parser.add_argument('-out', type=str, help='Output dir for tsv')
parser.add_argument('--batch', type=int, default=1024, help='Batching')
parser.add_argument('--verbose',action='store_true')

def main(args):
    logging.info('Loading Dataset')
    ds = ir_datasets.load("msmarco-passage/train/triples-small")
    logging.info('Intialising Collection')
    scorer = Mine(ds)

    cols = ['q', 'pos', 'neg']
    types = {col : str for col in cols}

    idx = [f for f in os.listdir(args.ids) if os.path.isfile(os.path.join(args.ids, f))]

    for id_file in idx:    
        logging.info(f'Processing {id_file}...')
        triples = pd.read_csv(os.path.join(args.ids, id_file), sep='\t', header=None, index_col=False, names=cols, dtype=types, chunksize=args.batch)

        frames = [scorer.get_batch(chunk) for chunk in triples]
        df = pd.concat(frames)

        df.to_csv(os.path.join(args.out, id_file), sep='\t', index=False, header=False)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Collating Triples--')
    main(args)