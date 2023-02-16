import argparse
import logging
import os

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--filter', type=str, nargs='*')
parser.add_argument('--verbose', action='store_true')

def main(args):

    files = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]
    if args.filter:
        files = [f for filter in args.filter for f in files if filter in f]

    for file in files:    
        new_name = 't5_' + file

        with open(os.path.join(args.sink, new_name), 'w') as fout:
            for line_num, line in enumerate(open(os.path.join(args.source, file))):
                query, positive_document, negative_document = line.strip().split('\t')
                fout.write(f'Query: {query} Document: {positive_document} Relevant:\ttrue\n')
                fout.write(f'Query: {query} Document: {negative_document} Relevant:\tfalse\n')
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    main(args)