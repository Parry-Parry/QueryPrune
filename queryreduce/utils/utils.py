import faiss
import numpy as np 

def time_output(diff : int):
    seconds = diff
    minutes = seconds / 60
    hours = minutes / 60 

    if hours > 1:
        return f'Completed search in {hours} hours'
    elif minutes > 1:
        return f'Completed search in {minutes} minutes'
    else:
        return f'Completed search in {seconds} seconds'

def to_device(index, ngpu : int):
    if ngpu == 1:
        res = faiss.StandardGpuResources() 
        index = faiss.index_cpu_to_gpu(res, 0, index)
    elif ngpu > 1:
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        return False 

    return True

'''
Function weights Embeddings by given hyperparameters

Assumes:
    * 3 flattened Embeddings on Dim 1 
'''

def weight(array : np.array, dim : int, alpha : float, beta : float, equal : bool) -> np.array:
    
    if equal: 
        faiss.normalize_L2(array)
        return array

    gamma = np.max(1 - alpha - beta, 0)

    array[:, :dim] = alpha * array[:, :dim]
    array[:, dim:2*dim] = beta * array[:, dim:2*dim]
    array[:, 2*dim:3*dim] = gamma * array[:, 2*dim:3*dim]
    faiss.normalize_L2(array)
    return array