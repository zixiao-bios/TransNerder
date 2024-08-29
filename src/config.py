import os
from dotenv import load_dotenv

from dataset import WMT_Dataset_en2zh

load_dotenv()

run_name = 'test_mac_02'

en_seq_len = 30
zh_seq_len = 45

epochs = 10
batch_size = 12
lr = 0.002

dataset = WMT_Dataset_en2zh(os.getenv('CSV_PATH', ''), zh_target_len=zh_seq_len, en_target_len=en_seq_len, batch_size=batch_size)
params = {
    'source_vocab_size': len(dataset.en_vocab), 
    'target_vocab_size': len(dataset.zh_vocab), 
    'source_embed_size': 300, 
    'target_embed_size': 200, 
    'source_seq_len': en_seq_len, 
    'target_seq_len': zh_seq_len, 
    'num_hiddens': 256, 
    'num_layers': 2
}
