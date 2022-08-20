from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import numpy as np 
import torch
import transformers
import argparse
import json

PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

parser = argparse.ArgumentParser()
parser.add_argument("exp_id", type=int, help="experiment ID")
parser.add_argument("seed", type=int, choices=[42,1,1234,123,10], help="random seed")
parser.add_argument("enc", type=str, choices=["bert", "roberta"], help="encoder")
parser.add_argument("model", type=str, choices=['bert-base-cased', "roberta-base"], help="model name")
args = parser.parse_args()

exp_id = args.exp_id
seed = args.seed
encoder = args.enc
model_name = args.model

max_length = 128
exp_dir = f"../experiments/{encoder}/{seed}/{exp_id}"

def prepare(params, samples):
    pass

def batcher(params, batch):  
    sent_batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    tokenizer_output = tokenizer.batch_encode_plus(sent_batch, max_length=max_length, pad_to_max_length=True)
    input_ids = torch.tensor(tokenizer_output["input_ids"])
    attention_mask = torch.tensor(tokenizer_output["attention_mask"])
    with torch.no_grad():
        last_hidden_state = model(input_ids, attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state.numpy()
    embeddings = last_hidden_state[:,0,:]
    return embeddings

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModel.from_pretrained(f"{exp_dir}/encoder")

params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10} # usepytorch=False
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params, batcher, prepare)
    transfer_tasks = ['Length', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
    with open(f"{exp_dir}/probe_results.json", "w") as write_file:
        json.dump(results, write_file, indent=4)
