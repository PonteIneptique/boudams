# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple

from .base import BaseSeq2SeqModel


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!

        self.rnn = nn.GRU(emb_dim, self.dec_hid_dim, bidirectional=True)

        self.fc = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len)-> Tuple[torch.Tensor, torch.Tensor]:
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((self.enc_hid_dim * 2) + self.dec_hid_dim, self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        #mask = [batch size, src sent len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention= [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.attention: Attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        # Since we added pack, we return attention
        return output, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module, BaseSeq2SeqModel):
    def __init__(self, encoder, decoder, device, pad_idx, sos_idx, eos_idx, out_max_sentence_length=150):
        super().__init__()

        self.out_max_sentence_length = out_max_sentence_length

        self.encoder = encoder
        self.decoder = decoder

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg=None, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        if trg is None:
            #src = src.permute(1, 0)
            teacher_forcing_ratio = 0  # Must be !
            out_max_len = self.out_max_sentence_length
            inference = True

            # In case we do not have a target fed to the forward function (basically when we tag)
            #   we create a target tensor filled with StartOfSentence tokens
            batch_size = src.shape[1]
            trg = torch.zeros(
                (self.out_max_sentence_length, batch_size)
            ).long().fill_(self.sos_idx).to(self.device)
        else:
            batch_size = src.shape[1]
            out_max_len = trg.shape[0]
            inference = False
            # first input to the decoder is the <sos> tokens


        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(out_max_len, batch_size, trg_vocab_size).to(self.device)
        #tensor to store attention
        attentions = torch.zeros(out_max_len, batch_size, src.shape[0]).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src, src_len)

        #first input to the decoder is the <sos> tokens
        output = trg[0, :]

        mask = self.create_mask(src)
        #mask = [batch size, src sent len]

        for t in range(1, out_max_len):
            output, hidden, attention = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention

            if random.random() < teacher_forcing_ratio:
                output = trg[t]
            else:
                output = output.max(1)[1]

            #print(output, outputs, self.eos_idx)
            if inference and output.item() == self.eos_idx:  # This does not take into account batch ! This fails with batches
                return outputs[:t], attentions[:t]

        return outputs, attentions


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

