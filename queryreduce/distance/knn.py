from typing import Tuple
import faiss
import numpy as np 

'''
KNN Implementation inspired by: https://github.com/CompVis/metric-learning-divide-and-conquer

Other functions are general examples taken from faiss codebase
'''

def get_centroids(x, d, K, max_iter : int = 300, max_points : int = 10000, ngpu : int = 1) -> dict:
    clustering = faiss.Clustering(d=d, k=K)
    clustering.niter = max_iter
    clustering.max_points_per_centroid = max_points

    if ngpu is not None:
        res = [faiss.StandardGpuResources() for i in ngpu]

        flat_config = []
        for i in ngpu:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = i
            flat_config.append(cfg)

        if len(ngpu) == 1:
            index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
        else:
            indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                       for i in range(len(ngpu))]
            index = faiss.IndexProxy()
            for sub_index in indexes:
                index.addIndex(sub_index)
    else:
        index = faiss.IndexFlatL2(d)

    clustering.train(x, index)
    centroids = faiss.vector_float_to_array(clustering.centroids)

    return centroids.reshape(K, d)

def compute_asignments(centroids : np.array, x : np.array) -> Tuple[np.array, np.array]:
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    D, I = index.search(x, 1)

    return D.ravel(), I.ravel()

def cluster_queries(triplets : list, K : int, max_iter : int = 300, max_points : int = 100000, ngpu : int = 1):
    x = np.array([trip.q.embed_obj for trip in triplets]) 
    centroids = get_centroids(x, x.shape[1], K, max_iter, max_points, ngpu)

    return compute_asignments(centroids, x)

class find_knn:
    def __init__(self, x, ngpu = None) -> None:
        x = np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32)
        self.d = x.shape[1]

        if ngpu is None:
            self.index = faiss.IndexFlatL2(self.d)
        else:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = ngpu

            flat_config = [cfg]
            resources = [faiss.StandardGpuResources()]
            self.index = faiss.GpuIndexFlatL2(resources[0], self.d, flat_config[0])

        self.index.add(x)
    
    def query(self, queries, k):
        return self.index.search(queries, k)