from collections import defaultdict
from queryreduce.utils.config import Text, TextSet
from sklearn.cluster import KMeans
import numpy as np 

def clustering(queries : TextSet, K : int, seed=8008) -> dict:
    x = queries.embeddings

    clustering = KMeans(n_clusters=K, random_state=seed).fit_predict(x)

    groups = defaultdict(list)

    for a, b, c in zip(x, clustering, queries.ids):
        groups[b].append(Text(c, a))
    
    return groups



