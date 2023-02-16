import argparse
import logging
import os
import pickle 
import pandas as pd 
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-experiment', type=str)

parser.add_argument('--filter', type=str, nargs='*')
parser.add_argument('--verbose', action='store_true')


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.partition(b' ')[0])

def main(args):
    files = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]
    if args.filter:
        files = [f for filter in args.filter for f in files if filter in f]

    for file in files:    
        dir = os.path.join(args.source, file)
        strip_name = file.strip('.tsv')

        out = subprocess.run([
            'python', '-m', 'colbert.train', '--amp', '--doc_maxlen', '180', '--mask-punctuation', '--maxsteps', str(wccount(dir)), '--triples', dir, '--root', args.root, '--experiment', args.experiment, '--run', strip_name
        ])
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Running Parameter Sweep for ColBERT--')
    main(args)
    
    

        