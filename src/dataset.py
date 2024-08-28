import pickle
from collections import Counter
import pandas as pd
import subprocess
import re


def zh_to_token(zh: str):
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]|[，。！？、；：“”（）《》〈〉—…‘’“”""\'\-—_/\[\]{}<>`~@#$%^&*+=|\\]', zh)

def en_to_token(en: str):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+|[.,!?;:\"\'\-(){}[\]<>`~#$%^&*+=|\\/_]", en)


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


class WMT_DatasetVocab(WMT_DatasetChunk):
    def __init__(self, csv_path, **kwargs):
        with open('data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
            self.zh_counter: Counter = data_dict['zh_counter']
            self.en_counter: Counter = data_dict['en_counter']
            super().__init__(csv_path, tot_lines=data_dict['line_num'], **kwargs)
