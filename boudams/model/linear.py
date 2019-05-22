# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSeq2SeqModel, pprint_2d, pprint_1d


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..trainer import Scorer


from .conv import Encoder


class LinearDecoder(nn.Module):
    """
    Simple Linear Decoder that outputs a probability distribution
    over the vocabulary
    Parameters
    ===========
    label_encoder : LabelEncoder
    in_features : int, input dimension
    """
    def __init__(self, enc_dim, out_dim, highway_layers=0, highway_act='relu'):
        super().__init__()

        # highway
        self.highway = None
        # decoder output
        self.decoder = nn.Linear(enc_dim, out_dim)

    def forward(self, enc_outs):
        if self.highway is not None:
            enc_outs = self.highway(enc_outs)
        linear_out = self.decoder(enc_outs)

        return linear_out


class Seq2Seq(BaseSeq2SeqModel):

    def __init__(
        self,
        encoder: Encoder, decoder: LinearDecoder,
        device: str,
        pad_idx: int, sos_idx: int, eos_idx: int,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, src, src_len, trg, **kwargs):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        # calculate z^u (encoder_conved) and e (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src sent len, emb dim]
        # encoder_combined = [batch size, src sent len, emb dim]

        output, = self.decoder(encoder_conved.view(1, 0, 2))

        return output

    def predict(self, src, src_len, label_encoder: "LabelEncoder") -> torch.Tensor:
        """ Predicts value for a given tensor

        :param src: tensor(batch size x sentence_length)
        :param src_len: tensor(batch size)
        :param label_encoder: Encoder
        :return: Reversed Batch
        """
        out = self(src, src_len, None, teacher_forcing_ratio=0)[0]
        logits = torch.argmax(out, 2)
        return label_encoder.reverse_batch(logits)

    def gradient(
        self,
        src, src_len, trg=None,
        scorer: "Scorer" = None, criterion=None,
        evaluate: bool = False,
        **kwargs
    ):
        """

        :param src: tensor(batch size x sentence length)
        :param src_len: tensor(batch_size)
        :param trg: tensor(batch_size x output_length)
        :param scorer: Scorer
        :param criterion: Loss System
        :param evaluate: Whether we are in eval mode
        :param kwargs:
        :return: tensor(batch_size x output_length)
        """

        output = self(src, src_len, trg[:, 1:])
        print(out.shape)

        # We register the current batch
        #  For this to work, we get ONLY the best score of output which mean we need to argmax
        #   at the second layer (base 0 I believe)
        # We basically get the best match at the output dim layer : the best character.

        # The prediction and ground truth batches NECESSARLY starts by "0" where
        #    0 is the SOS token. In order to have a score independant from hardcoded ints,
        #    we remove the first element of each sentence

        scorer.register_batch(
            torch.argmax(output, 2),
            trg[:, 1:]
        )

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        # About contiguous : https://stackoverflow.com/questions/48915810/pytorch-contiguous
        # Basically, elements of the tensor are spread over memory and to make it VERY simple, it's a bit like
        #   deepcopy.

        loss = criterion(
            output.contiguous().view(-1, output.shape[-1]),
            trg[:, 1:].contiguous().view(-1)
        )

        return loss
