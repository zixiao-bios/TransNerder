from dataset import Vocab

from config import en_counter, zh_counter, data_params, net_params

print('Load vocab...')
en_vocab = Vocab(en_counter)
zh_vocab = Vocab(zh_counter)

def get_params():
    data_params['zh_vocab'] = zh_vocab
    data_params['en_vocab'] = en_vocab

    net_params['source_vocab_size'] = len(en_vocab)
    net_params['target_vocab_size'] = len(zh_vocab)
    return data_params, net_params