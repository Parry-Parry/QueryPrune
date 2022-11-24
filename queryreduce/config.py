from typing import NamedTuple, Any, Union
from numpy import array


### Config Structures ###

class EmbedConfig(NamedTuple):
    tokenizer : Any 
    model : Any
    dataset : Any

class MarkovConfig(NamedTuple):
    alpha : Any
    beta : Any
    triples : Any
    distr : Any
    gpu : Any


