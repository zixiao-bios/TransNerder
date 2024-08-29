import pickle
from collections import Counter
import pandas as pd
import subprocess
import re
import html
import torch


def zh_to_token(zh: str):
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]|[，。！？、；：“”（）《》〈〉—…‘’“”""\'\-—_/\[\]{}<>`~@#$%^&*+=|\\]', zh)

def en_to_token(en: str):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+|[.,!?;:\"\'\-(){}[\]<>`~#$%^&*+=|\\/_]", en)

def make_vocab(counter: Counter, min_freq=400):
    idx_to_token = ['[pad]', '[sos]', '[eos]', '[unk]']
    for token, freq in counter.most_common():
        if freq >= min_freq:
            idx_to_token.append(token)
    
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    return idx_to_token, token_to_idx

class Vocab:
    def __init__(self, counter: Counter, min_freq=400):
        self.idx_to_token, self.token_to_idx = make_vocab(counter, min_freq)
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def tokens_to_idx(self, tokens: list[str]):
        return [self.token_to_idx.get(token, self.token_to_idx['[unk]']) for token in tokens]
    
    def idx_to_tokens(self, indices: list[int]):
        res = []
        for idx in indices:
            if idx < 0 or idx >= len(self.idx_to_token):
                res.append('[unk]')
            else:
                res.append(self.idx_to_token[idx])
        return res
    
    def idx_to_str(self, indices: list[int]) -> str:
        return ' '.join(self.idx_to_tokens(indices))
    
    def trim_tokens(self, tokens: list[str], target_len, add_sos=False, add_eos=False):
        if add_sos:
            tokens = ['[sos]'] + tokens
        
        trim_len = target_len - 1 if add_eos else target_len
        
        if len(tokens) > trim_len:
            tokens = tokens[:trim_len]
            
        valid_len = len(tokens)
        
        if add_eos:
            tokens.append('[eos]')
        
        tokens.extend(['[pad]'] * (target_len - len(tokens)))
        
        assert len(tokens) == target_len, f'len(tokens) = {len(tokens)} != target_len = {target_len}'
        return tokens, valid_len
    
    def trim_indices(self, indices: list[int], target_len, add_sos=False, add_eos=False):
        tokens = self.idx_to_tokens(indices)
        tokens, valid_len = self.trim_tokens(tokens, target_len, add_sos, add_eos)
        return self.tokens_to_idx(tokens), valid_len

class WMT_DatasetChunk:
    def __init__(self, csv_path, batch_size=100000, shuffle=False, tot_lines=None):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            raise NotImplementedError('Shuffle is not implemented yet.')
        
        if tot_lines is not None:
            self.tot_lines = tot_lines
        else:
            result = subprocess.run(['wc', '-l', csv_path], stdout=subprocess.PIPE)
            self.tot_lines = int(result.stdout.decode().split()[0]) - 1
        
        self.data_reader = pd.read_csv(csv_path, chunksize=batch_size)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        chunk = next(self.data_reader)
        zh_lines = chunk.iloc[:, 0].astype(str).tolist()
        en_lines = chunk.iloc[:, 1].astype(str).tolist()
        assert len(zh_lines) == len(en_lines), 'Length of zh and en should be the same.'
        return (zh_lines, en_lines)


class WMT_Dataset_en2zh(WMT_DatasetChunk):
    def __init__(self, csv_path, zh_target_len, en_target_len, **kwargs):
        with open('data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
            self.zh_counter: Counter = data_dict['zh_counter']
            self.en_counter: Counter = data_dict['en_counter']
            super().__init__(csv_path, tot_lines=data_dict['line_num'], **kwargs)
        
        self.zh_target_len = zh_target_len
        self.en_target_len = en_target_len
        
        self.zh_vocab = Vocab(self.zh_counter)
        self.en_vocab = Vocab(self.en_counter)
    
    def __next__(self): # type: ignore
        zh_lines, en_lines = super().__next__()
        
        zh_tokens = []
        zh_tokens_no_sos = []
        zh_valid_lens = []
        for line in zh_lines:
            tokens = zh_to_token(line)
            tokens_sos, _ = self.zh_vocab.trim_tokens(tokens, self.zh_target_len, add_sos=True, add_eos=True)
            zh_tokens.append(tokens_sos)
            
            tokens_no_sos, valid_len = self.zh_vocab.trim_tokens(tokens, self.zh_target_len, add_sos=False, add_eos=True)
            zh_tokens_no_sos.append(tokens_no_sos)
            zh_valid_lens.append(valid_len)
        
        en_tokens = []
        en_valid_lens = []
        for line in en_lines:
            tokens = en_to_token(html.unescape(line))
            tokens, valid_len = self.en_vocab.trim_tokens(tokens, self.en_target_len, add_sos=False, add_eos=True)
            en_tokens.append(tokens)
            en_valid_lens.append(valid_len)
        
        zh_input = [self.zh_vocab.tokens_to_idx(tokens) for tokens in zh_tokens]
        zh_target = [self.zh_vocab.tokens_to_idx(tokens) for tokens in zh_tokens_no_sos]
        en_input = [self.en_vocab.tokens_to_idx(tokens) for tokens in en_tokens]

        return (torch.tensor(zh_input), torch.tensor(en_input), torch.tensor(zh_target), torch.tensor(zh_valid_lens).reshape(-1, 1), torch.tensor(en_valid_lens).reshape(-1, 1), zh_tokens, en_tokens)
