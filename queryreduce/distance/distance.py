from collections import defaultdict
from queryreduce.utils.config import Text, TextSet
from sklearn.cluster import KMeans
import numpy as np 

def group(triplets : list, K : int, seed=8008) -> dict:
    x = [trip.q.embed_obj for trip in triplets]
    ids = [trip.q.id for trip in triplets]
    clustering = KMeans(n_clusters=K, random_state=seed).fit_predict(x)

    groups = defaultdict(list)
    for a, b in zip(clustering, triplets):
        groups[a].append(b)
    
    return groups