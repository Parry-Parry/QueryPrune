# Query Reduce
## Andrew Parry | UofG IR Group

### Aim
To effectively search a dataset for a subset that is highly representative of the original set. This implementation uses a Markov process to search for candidates conditioned on the previous candidate. This method requires a heuristic function which in this case is a weighted cosine similarity. 

### Limits
Current implementation is dependent on faiss so distance heuristics are effectively limited to the l2 norm and the inner product. The current implementation can be used for triples in a ranking or recommendation task e.g text triplets of the form <query, psg+, psg->. This will be expanded in future. The provided utilities will batch encode and concatenate embeddings into a format that can be searched by the Markov process. Implementations in this repo are refined from my other works.
