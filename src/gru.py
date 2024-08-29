import torch
from torch import nn


class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, seq_len, dropout=0, **kwargs):
        super(EncoderGRU, self).__init__(**kwargs)

        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout, bidirectional=False)
    
    def forward(self, X):
        assert X.ndim == 2 and X.shape[1] == self.seq_len, f'X.shape[1] = {X.shape[1]} != self.seq_len = {self.seq_len}'

        # X.shape = (batch_size, seq_len)
        X = self.embedding(X)
        # X.shape = (batch_size, seq_len, embed_size)
        X = X.permute(1, 0, 2)
        # X.shape = (seq_len, batch_size, embed_size)
        
        S, H = self.rnn(X)
        # S.shape = (seq_len, batch_size, num_hiddens)
        # H.shape = (num_layers, batch_size, num_hiddens)
        return S, H

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, seq_len, dropout=0, **kwargs):
        super(DecoderGRU, self).__init__(**kwargs)
        
        self.seq_len = seq_len
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def forward(self, X, H):
        assert X.ndim == 2 and X.shape[1] == self.seq_len, f'X.shape[1] = {X.shape[1]} != self.seq_len = {self.seq_len}'
        assert H.ndim == 3, f'H.ndim = {H.ndim} != 3'
        assert H.shape[0] == self.num_layers and H.shape[2] == self.num_hiddens, f'Wrong H.shape = {H.shape}'

        # X.shape = (batch_size, seq_len)
        X = self.embedding(X).permute(1, 0, 2)
        # X.shape = (seq_len, batch_size, embed_size)
        
        context = H[-1].repeat(X.shape[0], 1, 1)
        # context.shape = (seq_len, batch_size, num_hiddens)
        
        X  = torch.cat((X, context), 2)
        # X.shape = (seq_len, batch_size, embed_size + num_hiddens)
        
        O, H = self.rnn(X, H)
        # O.shape = (seq_len, batch_size, num_hiddens)
        O = self.dense(O).permute(1, 0, 2)
        # O.shape = (batch_size, seq_len, vocab_size)
        return O, H

class Seq2SeqGRU(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_embed_size, target_embed_size, source_seq_len, target_seq_len, num_hiddens, num_layers):
        super(Seq2SeqGRU, self).__init__()
        
        self.encoder = EncoderGRU(source_vocab_size, source_embed_size, num_hiddens, num_layers, source_seq_len)
        self.decoder = DecoderGRU(target_vocab_size, target_embed_size, num_hiddens, num_layers, target_seq_len)
    
    def forward(self, source, target):
        assert source.ndim == 2 and target.ndim == 2, f'source.ndim = {source.ndim}, target.ndim = {target.ndim}'
        assert source.shape[0] == target.shape[0], f'source.shape[0] = {source.shape[0]} != target.shape[0] = {target.shape[0]}'
        
        # source.shape = (batch_size, source_len)
        # target.shape = (batch_size, target_len)
        enc_o, H = self.encoder(source)
        O, _ = self.decoder(target, H)
        # O.shape = (batch_size, seq_len, target_vocab_size)
        return O
