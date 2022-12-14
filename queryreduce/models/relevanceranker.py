import numpy as np 
import faiss 
import pandas as pd

class ClusterEngine:
    def __init__(self, config) -> None:
        self.niter = config.niter
        self.nclust = config.nclust
        self.min = config.cmin 
        self.kmeans = None
    
    def query(self, x) -> np.array:
        assert self.kmeans is not None
        _, I = self.kmeans.index.search(x, 1)
        return I.ravel()

    def train(self, x) -> None:
        self.kmeans = faiss.Kmeans(x.shape[-1], self.nclust, niter=self.niter, verbose=False, spherical=True, min_points_per_centroid=self.min)
        self.kmeans.train(x)

class BM25scorer:
    def __init__(self, attr='text', index=None) -> None:
        import pyterrier as pt
        pt.init()
        self.attr = attr
        if index: self.scorer = pt.text.scorer(body_attr=attr, wmodel='BM25', background_index=index)
        else: self.scorer = pt.text.scorer(body_attr=attr, wmodel='BM25')

    def _convert_triple(self, df, focus):
        query_df = pd.DataFrame()
        query_df['qid'] = df['qid']
        query_df['query'] = df['query']
        query_df['docno'] = 'd1'
        query_df[self.attr] = df[focus]
        query_df['cluster_id'] = df['cluster_id']
        query_df['relative_index'] = df['relative_index']

        return query_df
    
    def score_set(self, df, focus):
        return self.scorer(self._convert_triple(df, focus))

    def score_pairs(self, df):
        scoring = df.sort_values(by=['diff'])['relative_index'].tolist()
        return scoring