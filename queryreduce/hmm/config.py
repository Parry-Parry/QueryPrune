from typing import Any, NamedTuple

class MarkovConfig(NamedTuple):
    triples : Any
    dim : int
    alpha : float
    beta : float
    equal : bool
    n : int
    k : int 
    store : str 
    built : bool

    
    