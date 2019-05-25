import torch
import torch.cuda

import os
import json
import tarfile
import logging
import re
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

    def annotate(self, texts: List[str], batch_size=32):
        self.model.eval()
        for n in range(0, len(texts), batch_size):
            batch = texts[n:n+batch_size]
            xs = [
                self.vocabulary.inp_to_numerical(self.vocabulary.prepare(s))
                for s in batch
            ]
            logging.info("Dealing with batch %s " % (int(n/batch_size)+1))
            tensor, sentence_length, order = self.vocabulary.pad_and_tensorize(
                    [x for x, _ in xs],
                    device=self.device,
                    padding=max(list(map(lambda x: x[1], xs)))
                )

            translations = self.model.predict(
                tensor, sentence_length, label_encoder=self.vocabulary
            )
            for index in range(len(translations)):
                yield "".join(translations[order.index(index)])

    def annotate_text(self, string, splitter=r"([⁊\W\d]+)", batch_size=32):
        splitter = re.compile(splitter)
        splits = splitter.split(string)

        tempList = splits + [""] * 2
        strings = ["".join(tempList[n:n + 2]) for n in range(0, len(splits), 2)]
        strings = list(filter(len, strings))

        treated = []
        max_size = self.out_max_sentence_length - 5
        for string in strings:
            if len(string) > max_size:
                treated.extend([
                    "".join(string[n:n + max_size])
                    for n in range(0, len(string), max_size)
                ])
            else:
                treated.append(string)

        for string in treated:
            if len(string) > max_size:
                print(len(string))
                print(string)

        yield from self.annotate(treated, batch_size=batch_size)
