import torch
import torch.nn as nn
import random

from .base import BaseSeq2SeqModel


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)  # no cell state!

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
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

        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # sent len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


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

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, src_len, trg=None, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        if trg is None:
            #src = src.permute(1, 0)
            teacher_forcing_ratio = 0
            out_max_len = self.out_max_sentence_length

            # In case we do not have a target fed to the forward function (basically when we tag)
            #   we create a target tensor filled with StartOfSentence tokens
            batch_size = src.shape[1]
            trg = torch.zeros(
                (self.out_max_sentence_length, batch_size)
            ).long().fill_(self.sos_idx).to(self.device)
            inference = True
        else:
            batch_size = src.shape[1]
            out_max_len = trg.shape[0]
            inference = False
            # first input to the decoder is the <sos> tokens


        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(out_max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        context = self.encoder(src, src_len)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # First input to the decoder is the <sos> tokens
        output = trg[0, :]

        for t in range(1, out_max_len):
            output, hidden = self.decoder(output, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force:
                output = trg[t]
            else:
                output = output.max(1)[1]

            if inference and output == self.eos_idx:  # This does not take into account batch ! This fails with batches
                return outputs[:t], None

        return outputs, None


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

