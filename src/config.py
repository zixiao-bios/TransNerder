import os
from dotenv import load_dotenv
import pandas as pd
import multiprocessing
import pickle
from collections import Counter

from dataset import WMT_Dataset_en2zh
from dataset import Vocab

load_dotenv()

run_name = 'test_mac'

en_seq_len = 30
zh_seq_len = 45

epochs = 10
batch_size = 12
lr = 0.002

dataset_num_process = 4

csv_path = os.getenv('CSV_PATH', '')

with open('data.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    zh_counter: Counter = data_dict['zh_counter']
    en_counter: Counter = data_dict['en_counter']
    tot_lines = data_dict['line_num']

en_vocab = Vocab(en_counter)
zh_vocab = Vocab(zh_counter)

data_params = {
    'csv_path': csv_path, 
    'tot_lines': tot_lines, 
    'zh_vocab': zh_vocab, 
    'en_vocab': en_vocab, 
    'zh_target_len': zh_seq_len, 
    'en_target_len': en_seq_len, 
    'batch_size': batch_size, 
    'num_process': dataset_num_process
}

net_params = {
    'source_vocab_size': len(en_vocab), 
    'target_vocab_size': len(zh_vocab), 
    'source_embed_size': 300, 
    'target_embed_size': 200, 
    'source_seq_len': en_seq_len, 
    'target_seq_len': zh_seq_len, 
    'num_hiddens': 256, 
    'num_layers': 2
}
