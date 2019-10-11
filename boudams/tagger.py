import torch
import torch.cuda

import os
import json
import tarfile
import logging
import regex as re
from typing import List

from boudams.model import linear
from boudams import utils

from .encoder import LabelEncoder


teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


class BoudamsTagger:
    def __init__(
            self,
            vocabulary: LabelEncoder,
            hidden_size: int = 256,
            enc_n_layers: int = 10,
            emb_enc_dim: int = 256,
            enc_hid_dim: int = None,
            enc_dropout: float = 0.5,
            enc_kernel_size: int = 3,
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

        self.device: str = device
        self.enc_hid_dim = self.dec_hid_dim = self.hidden_size = hidden_size

        if enc_hid_dim:
            self.enc_hid_dim: int = enc_hid_dim

        self.emb_enc_dim: int = emb_enc_dim
        self.enc_dropout: float = enc_dropout
        self.enc_kernel_size: int = enc_kernel_size
        self.enc_n_layers: int = enc_n_layers

        self.out_max_sentence_length: int = out_max_sentence_length
        self.system: str = system

        # Based on self.masked, decoder dimension can be drastically different
        self.dec_dim = len(self.vocabulary.itom)

        self.mask_token = self.vocabulary.mask_token

        seq2seq_shared_params = {
            "pad_idx": self.padtoken,
            "device": self.device,
            "out_max_sentence_length": self.out_max_sentence_length
        }

        if self.system.endswith("-lstm"):
            self.enc: linear.LSTMEncoder = linear.LinearLSTMEncoder(
                    self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                    n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                    dropout=self.enc_dropout
                )
            in_features = self.enc_hid_dim
        elif self.system.endswith("-gru"):
            self.enc: linear.BiGruEncoder = linear.BiGruEncoder(
                    self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                    n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                    dropout=self.enc_dropout
                )
            in_features = self.enc_hid_dim
        elif self.system.endswith("-conv-no-pos"):
            self.enc: linear.LinearEncoderCNNNoPos = linear.LinearEncoderCNNNoPos(
                    self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                    n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                    dropout=self.enc_dropout,
                    device=self.device,
                    kernel_size=self.enc_kernel_size
                )
            in_features = self.emb_enc_dim
            # This model does not need sentence length
            self.out_max_sentence_length = None
        else:
            self.enc: linear.CNNEncoder = linear.LinearEncoderCNN(
                    self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                    n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                    dropout=self.enc_dropout,
                    device=self.device,
                    kernel_size=self.enc_kernel_size,
                    max_sentence_len=self.out_max_sentence_length
                )
            in_features = self.emb_enc_dim

        self.dec: linear.LinearDecoder = linear.LinearDecoder(
            enc_dim=in_features, out_dim=len(self.vocabulary.mtoi)
        )
        self.model: linear.LinearSeq2Seq = linear.LinearSeq2Seq(
            self.enc, self.dec,
            pos="nopos" not in self.system,
            **seq2seq_shared_params
        ).to(device)
        self.init_weights = None

    @property
    def padtoken(self):
        return self.vocabulary.pad_token_index

    @property
    def settings(self):
        return {
            "enc_kernel_size": self.enc_kernel_size,
            "enc_n_layers": self.enc_n_layers,
            "hidden_size": self.hidden_size,
            "enc_hid_dim": self.enc_hid_dim,
            "emb_enc_dim": self.emb_enc_dim,
            "enc_dropout": self.enc_dropout,
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
                if device == "cpu":
                    obj.model.load_state_dict(torch.load(dictpath, map_location=device))
                else:
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
                tensor, sentence_length, label_encoder=self.vocabulary,
                override_src=[batch[order_id] for order_id in order]
            )
            for index in range(len(translations)):
                yield "".join(translations[order.index(index)])

    def annotate_text(self, string, splitter=r"([⁊\W\d]+)", batch_size=32):
        splitter = re.compile(splitter)
        splits = splitter.split(string)

        tempList = splits + [""] * 2
        strings = ["".join(tempList[n:n + 2]) for n in range(0, len(splits), 2)]
        strings = list(filter(len, strings))

        if self.out_max_sentence_length:
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
            strings = treated

        yield from self.annotate(strings, batch_size=batch_size)
