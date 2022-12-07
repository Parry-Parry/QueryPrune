from typing import Any, NamedTuple

class MarkovConfig(NamedTuple):
    triples : Any
    dim : int
    alpha : float
    beta : float
    equal : bool
    batch : int
    batch_type : str
    choice : str
    n : int
    store : str 
    nprobe : int
    ngpu : int

class AcceptanceConfig(NamedTuple):
    states : Any
    metric : str 
    sub : int 
    alpha : float 
    update : int 
    compare : str 
    gpus : int

class CentroidConfig(NamedTuple):
    states : Any
    sub : int 
    update : int 
    compare : str 
    threshold : str
    threshold_init : float
    gpus : int

    
    