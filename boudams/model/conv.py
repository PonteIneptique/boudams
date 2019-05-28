# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSeq2SeqModel, pprint_2d, pprint_1d


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..trainer import Scorer


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device: str = "cpu",
                 max_sentence_len: int = 100):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device
        self.max_sentence_len = max_sentence_len

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_sentence_len, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # create position tensor

        # pos = [src sent len, batch size] (Not what is documented)
        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = pos_embedded = [batch size, src sent len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, src sent len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, src sent len, hid dim]


        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, src sent len]

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2*hid dim, src sent len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src sent len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src sent len]

            # set conv_input to conved for next lo`op iteration
            conv_input = conved

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, src sent len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale

        # combined = [batch size, src sent len, emb dim]
        return conved, combined
