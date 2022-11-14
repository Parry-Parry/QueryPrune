from dataclasses import dataclass
from typing import NamedTuple, Any, Union
import torch
from numpy import array

class EmbedConfig(NamedTuple):
    tokenizer : Any 
    model : Any
    dataset : Any

class Embedding(NamedTuple):
    embed_obj : Any 
    id : Any

class Triplet(NamedTuple):
    q : Any
    d_pos : Any 
    d_neg : Any

