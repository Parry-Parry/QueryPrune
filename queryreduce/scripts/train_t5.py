import argparse
import logging
import os
import pickle
import time 
import pandas as pd 
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-experiment', type=str)
parser.add_argument('-epochs', type=str)

parser.add_argument('--batch', type=str, default='64')
parser.add_argument('--filter', type=str, nargs='*')
parser.add_argument('--verbose', action='store_true')

def main(args):
    if not os.path.exists(os.path.join(args.root, args.experiment)):
        logging.info('ROOT does not exist, creating...')
        os.mkdir(os.path.join(args.root, args.experiment))

    files = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]
    if args.filter:
        files = [f for filter in args.filter for f in files if filter in f]

    for file in files:    
        strip_name = file.strip('.tsv')
        start = time.time()
        out = subprocess.run([
            'python', '-m', 'pygaggle.run.finetune_monot5', '--triples_path', os.path.join(args.source, file), '--output_model_path', os.path.join(args.root, args.experiment, strip_name), '--save_every_n_steps', '5000', '--per_device_train_batch_size', args.batch, '--output_model_path', os.path.join(args.experiment, strip_name), '--epochs', args.epochs
        ])
        end = time.time() - start 
        with open(os.path.join(args.root, args.experiment, strip_name, 'time.pkl'), 'wb') as f:
            pickle.dump(end)
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Running Parameter Sweep for T5--')
    main(args)
    
    

        