from typing import Tuple
import faiss
from functools import partial
import numpy as np 

'''
Distance functions for Markov Process sample choice
'''

def init_penalty_distance(resource, alpha, beta, gamma):
    norm = partial(faiss.pairwise_distance_gpu, resource)
    def distance(x, xs):
        return alpha * norm(x[:, 0], xs[:, 0]) + beta * norm(x[:, 1], xs[:, 1]) + () * norm(x[:, 2], xs[:, 2]) - gamma * norm(xs[:, 1], xs[:, 2])

    return distance

def init_interpolated_distance(resource, alpha, beta, equal=False):
    gamma = 1 - alpha - beta
    if gamma < 0: gamma = 0
    if equal:
        alpha, beta, gamma = 1, 1, 1

    norm = partial(faiss.pairwise_distance_gpu, resource)
    def distance(x, xs):
        return alpha * norm(x[:, 0], xs[:, 0]) + beta * norm(x[:, 1], xs[:, 1]) + gamma * norm(x[:, 2], xs[:, 2])

    return distance




    