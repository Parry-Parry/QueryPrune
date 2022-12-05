from typing import Any, NamedTuple

class MarkovConfig(NamedTuple):
    triples : Any
    dim : int
    alpha : float
    beta : float
    equal : bool
    batch : int
    batch_type : str
    n : int
    store : str 
    nprobe : int
    ngpu : int


    
    