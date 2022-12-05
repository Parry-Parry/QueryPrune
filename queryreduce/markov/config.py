from typing import Any, NamedTuple

class MarkovConfig(NamedTuple):
    triples : Any
    dim : int
    alpha : float
    beta : float
    equal : bool
    batch : int
    n : int
    k : int 
    store : str 
    nprobe : int
    ngpu : int
    built : bool


    
    