import pickle
from collections import Counter
import pandas as pd
import re
import html
from typing import Callable
import torch
import multiprocessing
import logging
import numpy as np

from log import log_queue, init_subprocess_logging


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

def process_lines(lines: list[str], tokenizer: list[Callable], vocab: 'Vocab', target_len: int, add_sos=False, add_eos=True):
    """传入若干行文本，返回处理后的tokens和valid_len，支持手动设置是否添加sos和eos

    Args:
        line (str): 输入的一行文本
        tokenizer (Callable): 函数列表，将文本转换为tokens的一系列函数，按顺序调用
        vocab (Vocab): 文本的词典对象
        target_len (int): 目标长度，用于截取和填充
        add_sos (bool, optional): _description_. Defaults to False.
        add_eos (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[list[list[str]], list[int]]: tokens, valid_len. tokens是处理后的tokens列表，valid_len是有效长度（不包含pad, sos, eos的长度）
    """
    tokens_list: list[list[str]] = []
    valid_lens: list[int] = [] 
    for line in lines:
        for f in tokenizer:
            line = f(line)
        assert type(line) == list, f'type(line) = {type(line)}'
        tokens, valid_len = vocab.trim_tokens(line, target_len, add_sos=add_sos, add_eos=add_eos)
        tokens_list.append(tokens)
        valid_lens.append(valid_len)
    return tokens_list, valid_lens

def process_target_lines(lines: list[str], tokenizer: list[Callable], vocab: 'Vocab', target_len: int):
    tokens_list, valid_lens = process_lines(lines, tokenizer, vocab, target_len + 1, True, True)
    input = [vocab.tokens_to_idx(tokens[:-1]) for tokens in tokens_list]
    target = [vocab.tokens_to_idx(tokens[1:]) for tokens in tokens_list]
    return input, target, valid_lens

def process_source_lines(lines: list[str], tokenizer: list[Callable], vocab: 'Vocab', target_len: int):
    tokens_list, valid_lens = process_lines(lines, tokenizer, vocab, target_len, False, True)
    input = [vocab.tokens_to_idx(tokens) for tokens in tokens_list]
    return input, valid_lens

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
            
        valid_len = len(tokens) - int(add_sos)
        
        if add_eos:
            tokens.append('[eos]')
        
        tokens.extend(['[pad]'] * (target_len - len(tokens)))
        
        assert len(tokens) == target_len, f'len(tokens) = {len(tokens)} != target_len = {target_len}'
        return tokens, valid_len
    
    def trim_indices(self, indices: list[int], target_len, add_sos=False, add_eos=False):
        tokens = self.idx_to_tokens(indices)
        tokens, valid_len = self.trim_tokens(tokens, target_len, add_sos, add_eos)
        return self.tokens_to_idx(tokens), valid_len

class WMTChunk:
    def __init__(self, data_reader):
        self.data_reader = data_reader
        
    def __iter__(self):
        return self
    
    def __next__(self):
        chunk = next(self.data_reader)
        
        zh_lines = chunk.iloc[:, 0].astype(str).tolist()
        en_lines = chunk.iloc[:, 1].astype(str).tolist()
        assert len(zh_lines) == len(en_lines), 'Length of zh and en should be the same.'
        return (zh_lines, en_lines)

def _process_chunk(csv_path, start_row, num_rows, batch_size, zh_target_len, en_target_len, zh_vocab, en_vocab, queue: multiprocessing.Queue, log_queue: multiprocessing.Queue):
    init_subprocess_logging(log_queue)
    logger = logging.getLogger('dataset_subprocess')
    
    try:
        logger.info(f'Process chunk: start_row={start_row}, num_rows={num_rows}')
        data_reader = pd.read_csv(csv_path, skiprows=start_row, nrows=num_rows, chunksize=batch_size)
        chunk = WMTChunk(data_reader)
        
        for zh_lines, en_lines in chunk:
            zh_input, zh_target, zh_valid_lens = process_target_lines(zh_lines, [zh_to_token], zh_vocab, zh_target_len)
            en_input, en_valid_lens = process_source_lines(en_lines, [html.unescape, en_to_token], en_vocab, en_target_len)
            queue.put([np.array(zh_input), np.array(en_input), np.array(zh_target), np.array(zh_valid_lens).reshape(-1, 1), np.array(en_valid_lens).reshape(-1, 1)])

        queue.put(None)
        logger.info(f'Finished process chunk: start_row={start_row}, num_rows={num_rows}')
    except Exception as e:
        logger.error(e)
        queue.put(None)

class WMT_Dataset_en2zh():
    def __init__(self, csv_path, tot_lines, zh_vocab, en_vocab, zh_target_len, en_target_len, batch_size, num_process=8):
        print('Dataset init...')
        
        self.csv_path = csv_path
        self.tot_lines = tot_lines
        self.num_process = num_process
        self.batch_size = batch_size
        
        self.zh_target_len = zh_target_len
        self.en_target_len = en_target_len
        
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        
        self.processes = []
    
    def reset_porcesses(self):
        for p in self.processes:
            p.terminate()
            p.join()
        
        self.processes = []
        self.finished_process = 0
        self.res_queue = multiprocessing.Queue(maxsize=2*self.num_process)

        lines_pre_process = round(self.tot_lines / self.num_process)
        for i in range(self.num_process):
            start_row = i * lines_pre_process
            if i == self.num_process - 1:
                num_rows = self.tot_lines - start_row
            else:
                num_rows = lines_pre_process
            
            p = multiprocessing.Process(target=_process_chunk, args=(self.csv_path, start_row, num_rows, self.batch_size, self.zh_target_len, self.en_target_len, self.zh_vocab, self.en_vocab, self.res_queue, log_queue))
            self.processes.append(p)
    
    def start(self):
        for p in self.processes:
            p.start()
    
    def __iter__(self):
        self.reset_porcesses()
        self.start()
        return self
    
    def __next__(self): # type: ignore
        res = self.res_queue.get()
        if res is None:
            self.finished_process += 1
            if self.finished_process == self.num_process:
                raise StopIteration
            
            return self.__next__()
        
        torch_res = []
        for each in res:
            torch_res.append(torch.from_numpy(each))
        return torch_res

    def join(self):
        for p in self.processes:
            p.join()
    
    def terminate(self):
        for p in self.processes:
            p.terminate()
