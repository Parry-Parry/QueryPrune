import argparse
import logging
import os
import pickle
import time 
import pandas as pd 
import ir_datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW

import torch
torch.manual_seed(0)

_logger = ir_datasets.log.easy()
OUTPUTS = ['true', 'false']

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

    BATCH_SIZE = args.batch

    files = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]
    if args.filter:
        files = [f for filter in args.filter for f in files if filter in f]

    for file in files:    
        strip_name = file.strip('.tsv')
        sink = os.path.join(args.root, args.experiment, strip_name)
        if not os.path.exists(os.path.join(args.root, args.experiment, strip_name)):
            logging.info('EXP does not exist, creating...')
            os.mkdir(os.path.join(args.root, args.experiment, strip_name))

        cols = ['query', 'pid', 'nid']
        types = {col : str for col in cols}
        df = pd.read_csv(os.path.join(args.source, file), sep='\t', header=None, index_col=False, names=cols, dtype=types)

        def iter_train_samples():    
            while True:
                for row in df.itertuples():
                    yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                    yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]

        start = time.time()
        train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

        model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        epoch = 0

        while epoch < args.epochs:
            with _logger.pbar_raw(desc=f'train {epoch}', total=16384 // BATCH_SIZE) as pbar:
                model.train()
                total_loss = 0
                count = 0
                for _ in range(len(df) // BATCH_SIZE):
                    inp, out = [], []
                    for i in range(BATCH_SIZE):
                        i, o = next(train_iter)
                        inp.append(i)
                        out.append(o)
                    inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
                    out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()
                    loss = model(input_ids=inp_ids, labels=out_ids).loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss = loss.item()
                    count += 1
                    pbar.update(1)
                    pbar.set_postfix({'loss': total_loss/count})
            epoch += 1

        end = time.time() - start 
        model.save_pretrained(os.path.join(sink, 'model'))
        with open(os.path.join(sink, 'time.pkl'), 'wb') as f:
            pickle.dump(end)
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Running Parameter Sweep for T5--')
    main(args)






    

        