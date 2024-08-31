import os
from dotenv import load_dotenv
import pickle
from collections import Counter

print('Load config...')

load_dotenv()

run_name = 'test_mac_log'

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

data_params = {
    'csv_path': csv_path, 
    'tot_lines': tot_lines, 
    'zh_vocab': None, 
    'en_vocab': None, 
    'zh_target_len': zh_seq_len, 
    'en_target_len': en_seq_len, 
    'batch_size': batch_size, 
    'num_process': dataset_num_process
}

net_params = {
    'source_vocab_size': -1, 
    'target_vocab_size': -1, 
    'source_embed_size': 300, 
    'target_embed_size': 200, 
    'source_seq_len': en_seq_len, 
    'target_seq_len': zh_seq_len, 
    'num_hiddens': 256, 
    'num_layers': 2
}
