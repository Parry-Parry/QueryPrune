import numpy as np 
import faiss 
import logging 
import bz2

def _build_index(self, triples : np.array, k : int, store : str):
        assert store is not None
        ngpus = faiss.get_num_gpus()
        if ngpus < 1:
            logging.error("Error! Faiss Indexing Requires GPU, Exiting...")
            return 1

        logging.info('Building Index')

        faiss.normalize_L2(triples)
        quantiser = faiss.IndexFlatL2(self.prob_dim) 
        cpu_index = faiss.IndexIVFFlat(quantiser, self.prob_dim, k, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(triples)
        cpu_index.add(triples)

        return 0