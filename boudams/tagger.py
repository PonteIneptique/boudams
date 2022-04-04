import torch
import torch.cuda

import os
import json
import tarfile
import logging
import regex as re
from dataclasses import dataclass
from typing import List, Any, Optional, Dict, ClassVar, Tuple

from boudams.model import linear
from boudams import utils

from .encoder import LabelEncoder

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim


teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


class CrossEntropyLoss(pl.LightningModule):
    def __init__(self, pad_index):
        super(CrossEntropyLoss, self).__init__()
        self._pad_index = pad_index
        self._nn = nn.CrossEntropyLoss(ignore_index=self._pad_index)

    def forward(self, y, gt) -> torch.TensorType:
        return self._nn(y, gt)


@dataclass
class OptimizerParams:
    name: str = "Adams"
    kwargs: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        if self.name == "Adams":
            kwargs_required: List[str] = []
            defaults = {"lr": 1e-3}
            kwargs_maps: Dict[str, str] = {}
            cls = optim.Adam
        for param in kwargs_required:
            if not self.kwargs.get(param):
                raise Exception(f"{self.name} requires parameter {param}")
                # return False
        return True

    def get_optimizer(self, parameters: Optional = None) -> optim.Optimizer:
        self.validate()
        cls = None
        defaults = {}
        kwargs_maps = {}
        if self.name == "Adams":
            defaults = {"lr": 1e-3}
            kwargs_maps: Dict[str, str] = {}
            cls = optim.Adam
        kwargs = {}
        kwargs.update(defaults)
        for parameter in self.kwargs:
            kwargs[kwargs_maps.get(parameter, parameter)] = self.kwargs[parameter]
        if parameters:
            return cls(parameters, **kwargs)
        return cls(**kwargs)


class BoudamsTagger(pl.LightningModule):
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
            optimizer: OptimizerParams = OptimizerParams,
            system: str = "bi-gru",
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
        super(BoudamsTagger, self).__init__()

        self.vocabulary: LabelEncoder = vocabulary
        self.vocabulary_dimension: int = len(self.vocabulary)

        self.optimizer_params: OptimizerParams = optimizer
        self.optimizer_params.validate()

        # Parse params and sizes
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

        # Build the module
        self._build_nn()

        # ToDo: Allow for DiceLoss
        self._loss = CrossEntropyLoss(self.vocabulary.pad_token_index)

    def _build_nn(self):
        seq2seq_shared_params = {
            "pad_idx": self.padtoken,
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
        )
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
    def load(cls, fpath="./model.boudams_model", device=DEVICE):
        with tarfile.open(utils.ensure_ext(fpath, 'boudams_model'), 'r') as tar:
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))

            # load state_dict
            vocab = LabelEncoder.load(
                json.loads(utils.get_gzip_from_tar(tar, "vocabulary.json"))
            )

            obj = cls(vocabulary=vocab, **settings)
            obj.to(device)

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

    def forward(self, x: torch.TensorType, x_len: Optional[torch.TensorType] = None) -> Any:
        return self.model.forward(x, x_len)

    def training_step(self, batch, batch_idx):  # -> pl.utilities.types.STEP_OUTPUT:
        """ Runs training step on a batch

        :param batch: Batch of data, structure ((X, X_LEN), GT)
        :param batch_idx:
        :return: Loss (currently)
        """
        (x, x_len), gt = batch
        y = self(x, x_len)
        loss = self.loss(y=y, gt=gt)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """ Runs training step on a batch

        :param batch: Batch of data, structure (X, X_LEN)#, GT)
        :param batch_idx:
        :param dataloader_idx:
        :return: Loss (currently)
        """
        (x, x_len) = batch
        y = self(x, x_len)
        return y

    def validation_step(self, batch, batch_idx):
        (x, x_len), gt = batch
        y = self(x, x_len)
        loss = self.loss(y=y, gt=gt)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return self.opt

    def annotate(self, texts: List[str], batch_size=32, device: str = "cpu"):
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
                    device=device,
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

    def dump(self, fpath="model"):
        fpath += ".boudams_model"
        fpath = utils.ensure_ext(fpath, 'boudams_model', infix=None)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        os.makedirs(dirname, exist_ok=True)

        with tarfile.open(fpath, 'w') as tar:
            # serialize settings
            utils.add_gzip_to_tar(
                json.dumps(self.tagger.settings),
                'settings.json.zip',
                tar
            )
            # Serialize vocabulary
            utils.add_gzip_to_tar(
                self.tagger.vocabulary.dump(),
                'vocabulary.json',
                tar
            )

            # serialize field
            with utils.tmpfile() as tmppath:
                torch.save(self.tagger.model.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

        return fpath