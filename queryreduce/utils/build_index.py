import argparse 
import logging
from typing import Any, List
import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader
from sentence_transformers import SentenceTransformer
import pickle
import bz2

'''
Retrieve text associated with triplets of IDs, generate embeddings and return tuples
'''

class VectorFactory:
    def __init__(self, model : str, **kwargs) -> None:
        self.model = SentenceTransformer(model)

    def _batch_encode(self, txt : List[str]) -> np.array:
        return self.model.encode(txt, convert_to_numpy=True)
    def _batch_create(self, triples : pd.DataFrame):
        e_q = self._batch_encode(triples['qid'].to_list())
        e_pos = self._batch_encode(triples['pid+'].to_list())
        e_neg = self._batch_encode(triples['pid-'].to_list())

        batch = np.stack([e_q, e_pos, e_neg], axis=1)

        return batch.reshape((batch.shape[0], -1))

    def run(self, triples : TextFileReader, out : str, compresslevel=9):
        batches = [self._batch_create(chunk) for chunk in triples]
        with bz2.BZ2File(out, 'rb', compresslevel=compresslevel) as f:
            pickle.dump(np.stack(batches, axis=0), f)

parser = argparse.ArgumentParser(description='Construct embedding clusters from triplets of IDs')

parser.add_argument('-dataset', '-d', type=str, help='Directory of Triples tsv')
parser.add_argument('-model', type=str, help='Name of Model Checkpoint for indexing [Sentence Transformer Compatible]')
parser.add_argument('-batch_size', type=int, help='Batch Size')
parser.add_argument('-out', type=str, help='Output file [TSV]')


def main(args):
    cols = ['qid', 'pid+', 'pid-']
    types = {col : str for col in cols}
    logging.info('Reading Dataset...')
    
    iterator_df = pd.read_csv(args.dataset, sep='\t', header=None, index_col=False, names=cols, dtype=types, chunksize=args.batch_size)

    logging.info(f'Running Vector Factory on model {args.model} with batch size {args.batch_size}')
    factory = VectorFactory(args.model)
    factory.run(iterator_df, args.out)
    logging.info('Completed Successfully')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(parser.parse_args())