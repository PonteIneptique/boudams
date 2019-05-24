import torch
import torch.cuda

from torchtext.data import ReversibleField, BucketIterator

import os
import json
import tarfile
import logging
from typing import List, Tuple

from .model import gru, lstm, bidir, conv, linear
from . import utils

from .encoder import LabelEncoder, DatasetIterator


teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


class Seq2SeqTokenizer:
    def __init__(
            self,
            vocabulary: LabelEncoder,
            hidden_size: int = 256,
            enc_n_layers: int = 10, dec_n_layers: int = 10,
            emb_enc_dim: int = 256, emb_dec_dim: int = 256,
            enc_hid_dim: int = None, dec_hid_dim: int = None,
            enc_dropout: float = 0.5, dec_dropout: float = 0.5,
            enc_kernel_size: int = 3, dec_kernel_size: int = 3,
            out_max_sentence_length: int = 150,
            device: str = DEVICE, system: str = "bi-gru",
            **kwargs # RetroCompat
    ):
        """

        :param vocabulary:
        :param hidden_size:
        :param n_layers:
        :param emb_enc_dim:
        :param emb_dec_dim:
        :param max_length:
        :param device:
        """

        self.vocabulary: LabelEncoder = vocabulary
        self.vocabulary_dimension: int = len(self.vocabulary)
        self.masked: bool = self.vocabulary.masked

        self.device: str = device
        self.enc_hid_dim = self.dec_hid_dim = self.hidden_size = hidden_size

        if enc_hid_dim and dec_hid_dim:
            self.enc_hid_dim: int = enc_hid_dim
            self.dec_hid_dim: int = dec_hid_dim

        self.emb_enc_dim: int = emb_enc_dim
        self.emb_dec_dim: int = emb_dec_dim
        self.enc_dropout: float = enc_dropout
        self.dec_dropout: float = dec_dropout
        self.enc_kernel_size: int = enc_kernel_size
        self.dec_kernel_size: int = dec_kernel_size
        self.enc_n_layers: int = enc_n_layers
        self.dec_n_layers: int = dec_n_layers

        self.out_max_sentence_length: int = out_max_sentence_length
        self.system: str = system

        # Based on self.masked, decoder dimension can be drastically different
        self.dec_dim: int = self.vocabulary_dimension
        if self.masked:
            self.dec_dim = len(self.vocabulary.itom)

        self.mask_token = self.vocabulary.mask_token

        seq2seq_shared_params = {
            "pad_idx": self.padtoken,
            "sos_idx": self.sostoken,
            "eos_idx": self.eostoken,
            "device": self.device,
            "out_max_sentence_length": self.out_max_sentence_length
        }

        if self.system.startswith("linear"):
            self.enc: linear.CNNEncoder = linear.CNNEncoder(
                    self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                    n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                    dropout=self.enc_dropout,
                    device=self.device,
                    kernel_size=self.enc_kernel_size,
                    max_sentence_len=self.out_max_sentence_length
                )
            self.dec: linear.LinearDecoder = linear.LinearDecoder(
                enc_dim=self.emb_enc_dim, out_dim=len(self.vocabulary.mtoi)
            )
            self.model: linear.LinearSeq2Seq = linear.LinearSeq2Seq(
                self.enc, self.dec, **seq2seq_shared_params
            ).to(device)
            self.init_weights = None

        elif self.system == "conv":
            self.enc: gru.Encoder = conv.Encoder(
                self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                dropout=self.enc_dropout,
                device=self.device,
                kernel_size=self.enc_kernel_size,
                max_sentence_len=self.out_max_sentence_length
            )
            self.dec: gru.Decoder = conv.Decoder(
                output_dim=self.dec_dim, emb_dim=self.emb_dec_dim,
                hid_dim=self.dec_hid_dim, dropout=self.enc_dropout,
                device=self.device, pad_idx=self.padtoken, kernel_size=self.enc_kernel_size,
                n_layers=self.dec_n_layers, max_sentence_len=self.out_max_sentence_length
            )
            self.init_weights = None
            self.model: gru.Seq2Seq = conv.Seq2Seq(self.enc, self.dec, **seq2seq_shared_params).to(device)
        elif self.system == "gru":
            self.enc: gru.Encoder = gru.Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size,
                                                self.enc_dropout)
            self.dec: gru.Decoder = gru.Decoder(self.dec_dim, self.emb_dec_dim, self.hidden_size,
                                                self.dec_dropout)

            self.model: gru.Seq2Seq = gru.Seq2Seq(self.enc, self.dec, **seq2seq_shared_params).to(device)
            self.init_weights = gru.init_weights
        elif self.system == "bi-gru":
            self.enc: gru.Encoder = bidir.Encoder(
                self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                enc_hid_dim=self.enc_hid_dim, dec_hid_dim=self.dec_hid_dim, dropout=self.enc_dropout
            )
            self.attention: bidir.Attention = bidir.Attention(
                enc_hid_dim=self.enc_hid_dim, dec_hid_dim=self.dec_hid_dim
            )
            self.dec: gru.Decoder = bidir.Decoder(
                output_dim=self.dec_dim, emb_dim=self.emb_dec_dim,
                enc_hid_dim=self.enc_hid_dim, dec_hid_dim=self.dec_hid_dim, dropout=self.enc_dropout,
                attention=self.attention
            )
            self.init_weights = bidir.init_weights
            self.model: gru.Seq2Seq = bidir.Seq2Seq(self.enc, self.dec, **seq2seq_shared_params).to(device)
        else:
            self.enc: lstm.Encoder = lstm.Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size,
                                                  self.enc_n_layers, self.enc_dropout)
            self.dec: lstm.Decoder = lstm.Decoder(self.dec_dim, self.emb_dec_dim, self.hidden_size,
                                                  self.dec_n_layers, self.dec_dropout)
            self.init_weights = lstm.init_weights
            self.model: lstm.Seq2Seq = lstm.Seq2Seq(self.enc, self.dec, **seq2seq_shared_params).to(device)

    def to(self, device: str):
        # ToDo: This does not work, fix it
        self.device = device
        self.vocabulary.device = device

        if hasattr(self, "attention"):
            self.attention.to(device)
        self.enc.to(device)
        self.dec.to(device)
        self.model.to(device)

    @property
    def padtoken(self):
        return self.vocabulary.pad_token_index

    @property
    def sostoken(self):
        return self.vocabulary.init_token_index

    @property
    def eostoken(self):
        return self.vocabulary.eos_token_index

    def tag(self, iterator: BucketIterator):
        self.model.eval()
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            output, attention = self.model(
                src, src_len, trg=None,
                teacher_forcing_ratio=0
            )  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [Maximum Sentence Length, Number of Sentence in batch, Number of possible characters]
            _, ind = torch.topk(output, 1, dim=2)
            # ind = [Maximum Sentence Length, Number of Sentences in Batch, One Result]

            # output = output[1:].view(-1, output.shape[-1])

            yield ind.squeeze().permute(1, 0)

    @property
    def settings(self):
        return {
            "enc_kernel_size": self.enc_kernel_size,
            "dec_kernel_size": self.dec_kernel_size,
            "enc_n_layers": self.enc_n_layers,
            "dec_n_layers": self.dec_n_layers,
            "hidden_size": self.hidden_size,
            "enc_hid_dim": self.enc_hid_dim,
            "dec_hid_dim": self.dec_hid_dim,
            "emb_enc_dim": self.emb_enc_dim,
            "emb_dec_dim": self.emb_dec_dim,
            "enc_dropout": self.enc_dropout,
            "dec_dropout": self.dec_dropout,
            "out_max_sentence_length": self.out_max_sentence_length,
            "system": self.system
        }

    @classmethod
    def load(cls, fpath="./model.tar", device=DEVICE):
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))

            # load state_dict
            vocab = LabelEncoder.load(
                json.loads(utils.get_gzip_from_tar(tar, "vocabulary.json"))
            )
            settings.update({"device": device})

            obj = cls(vocabulary=vocab, **settings)

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
                obj.model.load_state_dict(torch.load(dictpath))

        obj.model.eval()

        return obj

    def annotate(self, texts: List[str]):
        self.model.eval()
        for sentence in texts:

            # it would be good at some point to keep and use order to batchify this
            tensor, sentence_length, _ = self.vocabulary.pad_and_tensorize(
                [self.vocabulary.inp_to_numerical(self.vocabulary.prepare(sentence))[0]],
                device=self.device,
                padding=self.out_max_sentence_length-len(sentence)
            )

            from .model.base import pprint_2d
            #pprint_2d(tensor.t())
            #print(sentence_length)

            logging.debug("Input Tensor {}".format(tensor.shape))
            logging.debug("Input Positions tensor {}".format(sentence_length.shape))

            translation = self.model.predict(
                tensor, sentence_length, label_encoder=self.vocabulary
            )

            yield "".join(translation[0])
