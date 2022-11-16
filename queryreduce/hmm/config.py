from typing import Any, Union, NamedTuple

class HMMconfig(NamedTuple):
    n_states : int 
    alpha : Any
    beta : Any
    index : Any
    triples : Any
    distr : Any