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
from .lstm import Encoder as LSTMEncoder
from .bidir import Encoder as BiGruEncoder


class LinearEncoderCNN(CNNEncoder):
    def forward(self, src, keep_pos=False):
        o, p = super(LinearEncoderCNN, self).forward(src)
        if keep_pos:
            return p
        return o


class LinearLSTMEncoder(LSTMEncoder):
    """ Linear
     version of the LSTMEncoder """


class LinearEncoderCNNNoPos(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device: str = "cpu"):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)

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

        # tok_embedded = pos_embedded = [batch size, src sent len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded)

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

        # combined = [batch size, src sent len, emb dim]
        return conved


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
        pad_idx: int,
        pos: bool = False,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder: LinearDecoder = decoder
        self.pos = pos

        self.pad_idx = pad_idx
        self.device = device

        # nll weight
        nll_weight = torch.ones(decoder.out_dim)
        nll_weight[pad_idx] = 0.
        self.register_buffer('nll_weight', nll_weight)

    def forward(self, src, src_len, trg=None, **kwargs):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        # calculate z^u (encoder_conved) and e (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings
        if isinstance(self.encoder, LinearEncoderCNN):
            second_step = self.encoder(src, keep_pos=True)
        elif isinstance(self.encoder, LinearEncoderCNNNoPos):
            second_step = self.encoder(src)
        elif isinstance(self.encoder, LinearLSTMEncoder):
            second_step, hidden, cell = self.encoder(src.t(), src_len)
            # -> tensor(sentence size, batch size, hid dim * n directions)
            second_step = second_step.transpose(1, 0)
            # -> tensor(batch size, sentence size, hid dim * n directions)
        elif isinstance(self.encoder, BiGruEncoder):
            second_step, hidden, cell = self.encoder(src.t(), src_len)
            # -> tensor(sentence size, batch size, hid dim * n directions)
            second_step = second_step.transpose(1, 0)
            # -> tensor(batch size, sentence size, hid dim * n directions)
        else:
            raise AttributeError("The encoder is not recognized.")

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
        return label_encoder.reverse_batch(
            logits,
            masked=override_src or src,
            ignore=(self.pad_idx, )
        )

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
