import torch
import torch.nn as nn
import random

from .base import BaseSeq2SeqModel


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(
            emb_dim, hid_dim, n_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self):
        return self.hid_dim * 2

    def forward(self, src):

        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        # packed_outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        output, hidden = self.rnn(embedded)

        return output, hidden

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

