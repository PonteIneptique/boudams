# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSeq2SeqModel, pprint_2d, pprint_1d


from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from ..trainer import Scorer


from .conv import Encoder as CNNEncoder


class LinearEncoderCNN(CNNEncoder):
    def forward(self, src):
        out = super(LinearEncoderCNN, self).forward(src)
        return out


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
        self.out_dim = out_dim
        # highway
        self.highway = None
        # decoder output
        self.decoder = nn.Linear(enc_dim, out_dim)

        self.relu = True

    def forward(self, enc_outs):
        if self.highway is not None:
            enc_outs = self.highway(enc_outs)

        linear_out = self.decoder(enc_outs)

        return linear_out


class LinearSeq2Seq(BaseSeq2SeqModel):
    masked_only = True

    def __init__(
        self,
        encoder: CNNEncoder, decoder: LinearDecoder,
        device: str,
        pad_idx: int, sos_idx: int, eos_idx: int,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder: LinearDecoder = decoder

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

        # nll weight
        nll_weight = torch.ones(decoder.out_dim)
        nll_weight[pad_idx] = 0.
        self.register_buffer('nll_weight', nll_weight)

    def forward(self, src, src_len, trg, **kwargs):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        # calculate z^u (encoder_conved) and e (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings
        if self.encoder is not None:
            _, second_step = self.encoder(src)
        else:
            second_step = src
        # encoder_conved = [batch size, src sent len, emb dim]
        # encoder_combined = [batch size, src sent len, emb dim]

        output = self.decoder(second_step)

        return output

    def predict(self, src, src_len, label_encoder: "LabelEncoder",
                override_src: Optional[List[str]] = None) -> torch.Tensor:
        """ Predicts value for a given tensor

        :param src: tensor(batch size x sentence_length)
        :param src_len: tensor(batch size)
        :param label_encoder: Encoder
        :return: Reversed Batch
        """
        out = self(src, src_len, None, teacher_forcing_ratio=0)
        logits = torch.argmax(out, 2)
        return label_encoder.reverse_batch(logits,
                                           masked=override_src or src,
                                           ignore=(self.pad_idx, self.eos_idx, self.sos_idx))

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
        output = self(src, src_len, trg)
        # -> tensor(batch_size * sentence_length)

        # We register the current batch
        #  For this to work, we get ONLY the best score of output which mean we need to argmax
        #   at the second layer (base 0 I believe)
        # We basically get the best match at the output dim layer : the best character.

        # The prediction and ground truth batches NECESSARLY starts by "0" where
        #    0 is the SOS token. In order to have a score independant from hardcoded ints,
        #    we remove the first element of each sentence

        scorer.register_batch(
            torch.argmax(output, 2),
            trg,
            src
        )

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        # About contiguous : https://stackoverflow.com/questions/48915810/pytorch-contiguous
        # Basically, elements of the tensor are spread over memory and to make it VERY simple, it's a bit like
        #   deepcopy.

        loss = criterion(
            output.view(-1, self.decoder.out_dim),
            trg.view(-1)
        )

        return loss

    def get_loss(self, preds, truths):
        """

        :param preds:
        :param truths:
        :return:
        """
        return F.cross_entropy(
            preds.view(-1, len(self.label_encoder)),
            truths.view(-1),
            weight=self.nll_weight, reduction="mean",
            ignore_index=self.label_encoder.get_pad()
        )
