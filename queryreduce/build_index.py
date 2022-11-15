from queryreduce.distance.distance import cluster_queries
from representations.triplets import * 
import argparse 
import ir_datasets

parser = argparse.ArgumentParser(description='Construct embedding clusters from triplets of IDs')

parser.add_argument('-dataset', '-d', type=str, help='Dataset from which to extract triplets')
parser.add_argument('-portion', '-p', type=int, default=100, help='How much of the dataset to parse')
parser.add_argument('-out', '-o', type=str, help='Output dir')
parser.add_argument('-neighbours', '-n', type=int, help='Number of Neighbours to prune')
parser.add_argument('-epsilon', '-e', type=float, help='Threshold distance')
parser.add_argument('-token_d', type=str, help='Document Tokenizer')
parser.add_argument('-pretrain', help='Train model on portion')


args = parser.parse_args()

dataset = ir_datasets.load(args.dataset)
out_dir = args.out 

model = None

config = EmbedConfig(
    tokenizer=None,
    model=model,
    dataset=dataset
)

lookup = EmbeddingWrapper(config)

if not dataset.has_docpairs():
    print(f"Dataset {args.dataset} does not have doc pairs to parse! EXITING...")
    exit 

triplets = [lookup.create_triplet(pair.query_id, pair.doc_id_a, pair.doc_id_b) for pair in dataset.doc_pairs_iter() ]

labels = cluster_queries(triplets, args.k)










