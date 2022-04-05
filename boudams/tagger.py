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
import torchmetrics

teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


class CrossEntropyLoss(pl.LightningModule):
    def __init__(self, pad_index, weights=None):
        super(CrossEntropyLoss, self).__init__()
        self._pad_index = pad_index
        self._nn = nn.CrossEntropyLoss(
            weight=weights, reduction="mean",
            ignore_index=self._pad_index
        )

    def forward(self, y, gt) -> torch.TensorType:
        return self._nn(y, gt)


@dataclass
class OptimizerParams:
    name: str = "Adams"
    kwargs: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        kwargs_required: List[str] = ["lr"]
        # Todo: Add Ranger
        #if self.name == "Adams":
        #    kwargs_required: List[str] = []
        for param in kwargs_required:
            if not self.kwargs.get(param):
                raise Exception(f"{self.name} requires parameter {param}")
                # return False
        return True

    def get_optimizer(
            self,
            model_parameters: Optional = None,
    ) -> Tuple[
        optim.Optimizer,
        optim.lr_scheduler.ReduceLROnPlateau
    ]:
        self.validate()
        cls = None
        defaults = {}
        kwargs_maps: Dict[str, str] = {}

        # Change kwargs and defaults
        if self.name == "Adams":
            cls = optim.Adam

        # Create kwargs
        kwargs = {}
        kwargs.update(defaults)
        for param in self.kwargs:
            kwargs[kwargs_maps.get(param, param)] = self.kwargs[param]

        if model_parameters:
            optimizer = cls(model_parameters, **kwargs)
        else:
            optimizer = cls(**kwargs)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=False,
            **self.scheduler
        )
        return optimizer, scheduler


class BoudamsTagger(pl.LightningModule):
    def __init__(
            self,
            vocabulary: LabelEncoder,
            metric_average: str = "macro",
            hidden_size: int = 256,
            enc_n_layers: int = 10,
            emb_enc_dim: int = 256,
            enc_hid_dim: int = None,
            enc_dropout: float = 0.5,
            enc_kernel_size: int = 3,
            out_max_sentence_length: int = 150,
            optimizer: Optional[OptimizerParams] = None,
            system: str = "bi-gru",
            have_metrics: bool = False,
            **kwargs # RetroCompat
    ):
        """

        :param vocabulary:
        :param hidden_size:
        :param n_layers:
        :param emb_enc_dim:
        :param emb_dec_dim:
        :param max_length:
        """
        super(BoudamsTagger, self).__init__()

        self.vocabulary: LabelEncoder = vocabulary
        self.vocabulary_dimension: int = len(self.vocabulary)

        self.optimizer_params: OptimizerParams = optimizer
        if self.optimizer_params:
            self.optimizer_params.validate()
            self.lr = self.optimizer_params.kwargs.get("lr")

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

        if self.optimizer_params:
            # ToDo: Allow for DiceLoss
            self.train_loss = CrossEntropyLoss(weights=self.model.nll_weight,
                                               pad_index=self.vocabulary.pad_token_index)
            self.val_loss = CrossEntropyLoss(weights=self.model.nll_weight,
                                             pad_index=self.vocabulary.pad_token_index)

        if metric_average not in {"micro", "macro"}:
            raise ValueError("`metric_average` can only be `micro` or `macro`")

        if self.optimizer_params or have_metrics:
            # Metrics
            for step in ["val", "test"]:
                self.add_metrics(step, metric_average=metric_average)

    def add_metrics(self, prefix, metric_average):
        metrics_params = dict(
            average=metric_average,
            num_classes=self.vocabulary.mask_count,
            ignore_index=self.vocabulary.pad_token_index
        )
        setattr(self, f"{prefix}_acc", torchmetrics.Accuracy(**metrics_params, subset_accuracy=True))
        setattr(self, f"{prefix}_f1", torchmetrics.F1Score(**metrics_params))
        setattr(self, f"{prefix}_pre", torchmetrics.Precision(**metrics_params))
        setattr(self, f"{prefix}_rec", torchmetrics.Recall(**metrics_params))
        setattr(self, f"{prefix}_stats", torchmetrics.StatScores(
            num_classes=self.vocabulary.mask_count,
            ignore_index=self.vocabulary.pad_token_index,
            multiclass=True
        ))

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
            in_features = self.enc.output_dim
        elif self.system.endswith("-gru"):
            self.enc: linear.BiGruEncoder = linear.BiGruEncoder(
                    self.vocabulary_dimension, emb_dim=self.emb_enc_dim,
                    n_layers=self.enc_n_layers, hid_dim=self.enc_hid_dim,
                    dropout=self.enc_dropout
                )
            in_features = self.enc.output_dim
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
        self.model: linear.MainModule = linear.MainModule(
            self.enc, self.dec,
            pos="nopos" not in self.system,
            **seq2seq_shared_params
        )

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
    def load(cls, fpath="./model.boudams_model"):
        with tarfile.open(utils.ensure_ext(fpath, 'boudams_model'), 'r') as tar:
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))

            # load state_dict
            vocab = LabelEncoder.load(
                json.loads(utils.get_gzip_from_tar(tar, "vocabulary.json"))
            )

            obj = cls(vocabulary=vocab, **settings)

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
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
        x, x_len, gt = batch
        y = self(x, x_len)
        loss = self.train_loss(*self._view_y_gt(y=y, gt=gt))
        self.log(f"loss", loss, batch_size=gt.shape[0], prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """ Runs training step on a batch

        :param batch: Batch of data, structure (X, X_LEN)#, GT)
        :param batch_idx:
        :param dataloader_idx:
        :return: Loss (currently)
        """
        x, x_len = batch
        y = self(x, x_len)
        return y

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        x, y, gt, _ = self._eval_step(batch, batch_idx, prefix="test")
        matrix = torchmetrics.functional.confusion_matrix(
            y, gt,
            num_classes=self.vocabulary.mask_count
        )
        return {"confusion_matrix": matrix}

    def test_epoch_end(self, outputs) -> None:
        confusion_matrix = [out["confusion_matrix"] for out in outputs]
        confusion_matrix = torch.stack(confusion_matrix, dim=-1).sum(dim=-1)
        # ToDo: Find a better way than assigning a value on tagger
        self.confusion_matrix = confusion_matrix
        self.log("test_wer", self._computer_wer(confusion_matrix), on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        stats_score = [out_dicts["confusion_matrix"] for (x, y, gt, out_dicts) in outputs]
        stats_score = torch.stack(stats_score, dim=-1).sum(dim=-1)
        self.log("val_wer", self._computer_wer(stats_score), on_epoch=True, prog_bar=True)

    def _computer_wer(self, confusion_matrix):
        indexes = torch.tensor([
            i
            for i in range(self.vocabulary.mask_count)
            if i != self.vocabulary.pad_token_index
        ]).type_as(confusion_matrix)
        clean_matrix = confusion_matrix[indexes][:, indexes]

        nb_space_gt = (
            clean_matrix[self.vocabulary.space_token_index].sum() +
            clean_matrix[:, self.vocabulary.space_token_index].sum() -
            clean_matrix[self.vocabulary.space_token_index, self.vocabulary.space_token_index]
        )

        nb_missed_space = clean_matrix.sum() - torch.diagonal(clean_matrix, 0).sum()
        return nb_missed_space / nb_space_gt

    def _eval_step(self, batch, batch_idx, prefix: str):
        x, x_len, gt = batch
        y = self(x, x_len)
        batch_size = gt.shape[0]
        y, gt = self._view_y_gt(y=y, gt=gt)

        # Remove manually index which are supposed to be pad...
        index_of_non_pads = (gt != self.vocabulary.pad_token_index).nonzero(as_tuple=False)

        if prefix != "test":
            loss = getattr(self, f"{prefix}_loss")(y=y, gt=gt)
            self.log(f"{prefix}_loss", loss, batch_size=batch_size, prog_bar=True)

        y = y[index_of_non_pads, :]
        gt = gt[index_of_non_pads]

        # for normal metrics, we simplify
        y = torch.argmax(y, -1)
        acc = getattr(self, f"{prefix}_acc")(y, gt)
        f1 = getattr(self, f"{prefix}_f1")(y, gt)
        rec = getattr(self, f"{prefix}_rec")(y, gt)
        pre = getattr(self, f"{prefix}_pre")(y, gt)
        matrix = torchmetrics.functional.confusion_matrix(
            y, gt,
            num_classes=self.vocabulary.mask_count
        )

        self.log(f"{prefix}_acc", acc, batch_size=batch_size, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_f1", f1, batch_size=batch_size, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_rec", rec, batch_size=batch_size, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_pre", pre, batch_size=batch_size, on_epoch=True, prog_bar=True)
        return x, y, gt, {"confusion_matrix": matrix}

    def _view_y_gt(self, y, gt):
        return y.view(-1, self.model.decoder.out_dim), gt.view(-1)

    def configure_optimizers(self):
        optimizer, scheduler = self.optimizer_params.get_optimizer(self.parameters())
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "monitor": "loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            ]
        )

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
                    padding=max(list(map(lambda x: x[1], xs)))
                )

            if device != "cpu":
                tensor, sentence_length = tensor.to(device), sentence_length.to(device)

            translations = self.model.predict(
                tensor, sentence_length, label_encoder=self.vocabulary,
                override_src=[batch[order_id] for order_id in order]
            )
            for index in range(len(translations)):
                yield "".join(translations[order.index(index)])

    def annotate_text(self, string, splitter=r"([⁊\W\d]+)", batch_size=32, device: str = "cpu"):
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

        yield from self.annotate(strings, batch_size=batch_size, device=device)

    def dump(self, fpath="model"):
        fpath += ".boudams_model"
        fpath = utils.ensure_ext(fpath, 'boudams_model', infix=None)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with tarfile.open(fpath, 'w') as tar:
            # serialize settings
            utils.add_gzip_to_tar(
                json.dumps(self.settings),
                'settings.json.zip',
                tar
            )
            # Serialize vocabulary
            utils.add_gzip_to_tar(
                self.vocabulary.dump(),
                'vocabulary.json',
                tar
            )

            # serialize field
            with utils.tmpfile() as tmppath:
                torch.save(self.model.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

        return fpath
