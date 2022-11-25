from typing import Tuple
import faiss
from functools import partial
import numpy as np 


'''
Distance functions for Markov Process sample choice
'''

def init_interpolated_distance(resource, alpha, beta, equal=False):
    gamma = 1 - alpha - beta
    if gamma < 0: gamma = 0
    if equal:
        alpha, beta, gamma = 1, 1, 1

    norm = partial(faiss.pairwise_distance_gpu, resource)
    def distance(x, xs):
        return alpha * norm(x[:, 0], xs[:, 0]) + beta * norm(x[:, 1], xs[:, 1]) + gamma * norm(x[:, 2], xs[:, 2])

    return distance

def init_interpolated_similarity_exhaustive(resource, alpha, beta, equal=False, METRIC=None):
    if not METRIC:
        METRIC = faiss.METRIC_INNER_PRODUCT

    gamma = 1 - alpha - beta
    if gamma < 0: gamma = 0
    if equal:
        alpha, beta, gamma = 1, 1, 1

    norm = lambda x : faiss.normalize_L2(x)
    dist = lambda x, y : faiss.pairwise_distance_gpu(resource, norm(x), norm(y), metric=METRIC) 

    def distance(x, xs):
        norm(x)
        return dist(x, xs)
    return distance







    