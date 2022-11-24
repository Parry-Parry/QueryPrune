from queryreduce.distance import cluster_queries
from representations.triplets import * 
import argparse 
import ir_datasets
import logging

parser = argparse.ArgumentParser(description='Construct embedding clusters from triplets of IDs')

parser.add_argument('-dataset', '-d', type=str, help='Directory of Triples tsv')
parser.add_argument('-model', type=str, help='Name of Model Checkpoint for indexing [Sentence Transformer Compatible]')
parser.add_argument('-batch_size', type=int, help='Batch Size')
parser.add_argument('-out', type=str, help='Output file')


def main(args):
    cols = ['qid', 'pid+', 'pid-']
    types = {str for col in cols}
    logging.info('Reading Dataset...')
    with open(args.dataset, 'r') as f:
        iterator_df = pd.read_csv(f, sep='\t', header=None, index_col=False, names=cols, dtype=types, chunksize=args.batch_size)

    logging.info('Running Vector Factory')
    factory = VectorFactory(args.model)
    factory.run(iterator_df, args.out)
    logging.info('Completed Successfully')
    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(parser.parse_args())










