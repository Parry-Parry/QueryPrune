import argparse
import logging
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Generate subsets of any array')

parser.add_argument('-source', type=str, help='Dataset Source')
parser.add_argument('-samples', type=int, help='How many samples', nargs='+')
parser.add_argument('-out', type=str, help='Output dir for tsv')

parser.add_argument('--verbose', action='store_true')

def main(args):
    cols = ['qid', 'pid+', 'pid-']
    types = {col : str for col in cols}
   
    df = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    for num in args.samples:
        sub_df = df.sample(n=num)
        sub_df.to_csv(os.path.join(args.out, f'triples.{num}.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    main(args)