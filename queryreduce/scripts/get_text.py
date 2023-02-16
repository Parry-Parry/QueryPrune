import argparse
import logging
import os
import pandas as pd
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-triples', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--filter', type=str, nargs='*')
parser.add_argument('--verbose', action='store_true')

def main(args):

    files = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]
    if args.filter:
        files = [f for filter in args.filter for f in files if filter in f]

    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    
    df = pd.read_csv(args.triples, sep='\t', header=None, index_col=False, names=cols, dtype=types)


    for file in files:    
        strip_name = file.strip('.pkl')
        with open(os.path.join(args.source, file), 'rb') as f:
            idx, _ = pickle.load(f)

        tmp = df.loc[idx]
        tmp.to_csv(os.path.join(args.sink, f'{strip_name}.tsv'), sep='\t', index=False, header=False)
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    main(args)