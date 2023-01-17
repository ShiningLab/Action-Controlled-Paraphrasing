#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import math
# public
import torch
import torch.nn as nn


class BiLSTMRNNEncoder(nn.Module):
    """Bidirectional RNN Enocder with Long Short Term Memory (LSTM)"""
    def __init__(self, embedding_dim, hidden_size, num_layers, max_seq_len, dropout_rate, proj=False):
        super(BiLSTMRNNEncoder, self).__init__()
        # bos + text + eos
        self.total_length = max_seq_len + 2
        self.proj = proj
        self.bi_lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        if self.proj:
            self.h_linear = torch.nn.Linear(num_layers*2*hidden_size, 2*hidden_size)
            self.c_linear = torch.nn.Linear(num_layers*2*hidden_size, 2*hidden_size)

    def forward(self, xs, x_lens):
        # xs: batch_size, src_seq_len, embedding_size
        # x_lens: batch_size
        xs = nn.utils.rnn.pack_padded_sequence(
            input=xs, 
            lengths=x_lens, 
            batch_first=True, 
            enforce_sorted=False)
        # h, c: 2*num_layers, batch_size, hidden_size
        xs, (h, c) = self.bi_lstm(xs)
        xs, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=xs
            , batch_first=True
            , total_length=self.total_length)
        # batch_size, src_seq_len, hidden_size
        xs = self.dropout(xs)
        if self.proj:
            # h, c: 1, batch_size, num_layer * hidden_size
            h = torch.unsqueeze(torch.cat(torch.unbind(h, 0), 1), 0)
            c = torch.unsqueeze(torch.cat(torch.unbind(c, 0), 1), 0)
            # h, c: 1, batch_size, hidden_size
            h, c = self.h_linear(h), self.c_linear(c)
        else:
            # 2, batch_size, hidden_size
            h = torch.stack((h[-2,:,:], h[-1,:,:]))
            c = torch.stack((c[-2,:,:], c[-1,:,:]))
        return xs, (h, c)


class PositionalEncoding(nn.Module):
    """Positional Encoder for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)