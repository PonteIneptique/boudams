import torch
import torch.cuda

import os
import json
import tarfile
import logging
import regex as re
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Any, Optional, Dict, ClassVar, Tuple

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from ranger import Ranger
import torchmetrics

from boudams.utils import improvement_on_min_or_max
from boudams.modules import *
from boudams import utils
from boudams.encoder import LabelEncoder

teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


_re_embedding = re.compile(r"E(\d+)")
_re_conv = re.compile(r"C(\d+),(\d+)(?:,(\d+))?")
_re_sequential_conv = re.compile(r"CS(s)?(\d+),(\d+),(\d+)(?:,Do(0?\.\d+))")
_re_pos = re.compile(r"P(l)?")
_re_bilstm = re.compile(r"L(\d+),(\d+)")
_re_bigru = re.compile(r"G(\d+),(\d+)")
_re_linear = re.compile(r"Li(\d+)")
_re_dropout = re.compile(r"Do(0?\.\d+)")


def _map_params(iterable):
    def weird_float(x):
        if x.startswith("."):
            return x[1:].isnumeric()
        return x.isnumeric()

    def eval_weird(x):
        if x.startswith("."):
            return float(f"0{x}")
        else:
            return eval(x)

    return (eval_weird(x) if x and weird_float(x) else x for x in iterable)


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
            monitored_metric: str = None,
            model_parameters: Optional = None
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
        elif self.name == "Ranger":
            cls = Ranger

        # Create kwargs
        kwargs = {}
        kwargs.update(defaults)
        for param in self.kwargs:
            kwargs[kwargs_maps.get(param, param)] = self.kwargs[param]

        if model_parameters:
            optimizer = cls(model_parameters, **kwargs)
        else:
            optimizer = cls(**kwargs)

        #scheduler = None
        #if self.name != "Ranger21":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=improvement_on_min_or_max(monitored_metric),
            verbose=False,
            **self.scheduler
        )
        return optimizer, scheduler


class ArchitectureStringError(ValueError):
    """ Error raised with wrong architecture string"""


def parse_architecture(
   string: str,
   maximum_sentence_size: Optional[int] = None
) -> Tuple[int, nn.ModuleDict, int]:
    """ Returns an embedding dimension and a module list

    >>> BoudamsTagger.parse_architecture("[E200 C5,10]")

    Should result in `(200, nn.ModuleList([Conv(200, out_filters=10, filter_size=5)]))`

    >>> BoudamsTagger.parse_architecture("[E200 C5,10,2]")
    Should result in `(200, nn.ModuleList([Conv(200, out_filters=10, filter_size=5, padding_size=2)]))`
    """
    if string[0] != "[" and string[-1] != "]":
        raise ArchitectureStringError("Architectures need top be encapsulated in [ ]")
    string = string[1:-1]
    modules: List[ModelWrapper] = []
    names: List[str] = []
    emb_dim = 0
    last_dim: int = 0
    for idx, module in enumerate(string.split()):
        if idx == 0:
            if _re_embedding.match(module):
                emb_dim = eval(_re_embedding.match(module).group(1))
                last_dim = emb_dim
            else:
                raise ArchitectureStringError("First module needs to be an embedding module. Start with [E<d>...]"
                                              " where <d> is a dimension, such as [E200")
        elif _re_embedding.match(module):
            raise ArchitectureStringError("You can't have embeddings after the first module")
        elif _re_conv.match(module):
            ngram, filter_dim, padding = _map_params(_re_conv.match(module).groups())
            modules.append(Conv(
                input_dim=last_dim,
                out_filters=filter_dim,
                filter_size=ngram,
                **(dict(padding_size=padding) if padding else {})
            ))
        elif _re_sequential_conv.match(module):
            use_sum, ngram, filter_dim, layers, drop = _map_params(_re_sequential_conv.match(module).groups())
            modules.append(SequentialConv(
                input_dim=last_dim,
                filter_dim=filter_dim,
                filter_size=ngram,
                n_layers=layers,
                dropout=drop,
                use_sum=use_sum or ""
            ))
        elif _re_pos.match(module):
            activation, = _map_params(_re_pos.match(module).groups())
            modules.append(PosEmbedding(
                input_dim=last_dim,
                maximum_sentence_size=maximum_sentence_size,
                activation=activation
            ))
        elif _re_bilstm.match(module):
            hidden_dim, layers = _map_params(_re_bilstm.match(module).groups())
            modules.append(BiLSTM(
                input_dim=last_dim,
                hidden_dim=hidden_dim,
                layers=layers
            ))
        elif _re_bigru.match(module):
            hidden_dim, layers = _map_params(_re_bigru.match(module).groups())
            modules.append(BiGru(
                input_dim=last_dim,
                hidden_dim=hidden_dim,
                layers=layers
            ))
        elif _re_linear.match(module):
            dim, = _map_params(_re_linear.match(module).groups())
            modules.append(Linear(
                input_dim=last_dim,
                output_dim=dim
            ))
        elif _re_dropout.match(module):
            rate, = _map_params(_re_dropout.match(module).groups())

            modules.append(Dropout(
                input_dim=last_dim,
                rate=rate
            ))
        else:
            raise ArchitectureStringError(f"Unknown `{module}` architecture")

        if len(modules):
            last_dim = modules[-1].output_dim
            names.append(module.replace(".", "_"))

    return emb_dim, nn.ModuleDict(OrderedDict(zip(names, modules))), last_dim


class BoudamsTagger(pl.LightningModule):

    def __init__(
            self,
            vocabulary: LabelEncoder,
            architecture: str = "[E256 G256,2 D.3]",
            metric_average: str = "macro",
            maximum_sentence_size: int = 150,
            optimizer: Optional[OptimizerParams] = None,
            have_metrics: bool = False
    ):
        """

        :param vocabulary:
        :param architecture:
        :param metric_average:
        :param maximum_sentence_size:
        :param optimizer:
        :param have_metrics:
        """
        super(BoudamsTagger, self).__init__()

        self.vocabulary: LabelEncoder = vocabulary
        self.vocabulary_dimension: int = len(self.vocabulary)

        self.optimizer_params: OptimizerParams = optimizer
        if self.optimizer_params:
            self.optimizer_params.validate()
            self.lr = self.optimizer_params.kwargs.get("lr")

        self._nb_classes = len(self.vocabulary.itom)

        # Parse params and sizes
        self._architecture: str = architecture
        _emb_dims, sequence, last_dim = parse_architecture(
            architecture,
            maximum_sentence_size=maximum_sentence_size
        )
        self._maximum_sentence_size: Optional[int] = maximum_sentence_size
        self._emb_dims = _emb_dims
        self._module_dict: nn.ModuleDict = sequence
        # Based on self.masked, decoder dimension can be drastically different
        self._embedder = nn.Embedding(self.vocabulary_dimension, self._emb_dims)
        self._classifier = nn.Linear(last_dim, self._nb_classes)

        if self.optimizer_params:
            # ToDo: Allow for DiceLoss
            # Needed when loading dict
            nll_weight = torch.ones(self._nb_classes)
            nll_weight[vocabulary.pad_token_index] = 0.
            self.register_buffer('nll_weight', nll_weight)

            self.train_loss = CrossEntropyLoss(weights=self.nll_weight,
                                               pad_index=self.vocabulary.pad_token_index)
            self.val_loss = CrossEntropyLoss(weights=self.nll_weight,
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

    @property
    def padtoken(self):
        return self.vocabulary.pad_token_index

    @property
    def settings(self):
        return {
            "architecture": self._architecture,
            "maximum_sentence_size": self._maximum_sentence_size
        }

    def forward(self, x: torch.TensorType, x_len: Optional[torch.TensorType] = None) -> torch.Tensor:
        after_seq_out = self._embedder(x)
        for module in self._module_dict.values():
            after_seq_out, x_len = module(after_seq_out, x_len)
        return self._classifier(after_seq_out)

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
        return self.vocabulary.mode.computer_wer(confusion_matrix)

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
        return y.view(-1, self._nb_classes), gt.view(-1)

    def configure_optimizers(self):
        optimizer, scheduler = self.optimizer_params.get_optimizer(
            model_parameters=self.parameters(),
            monitored_metric="loss"
        )
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "loss",
                    "frequency": 1
                }
            ]
        )

    def annotate(self, texts: List[str], batch_size=32, device: str = "cpu"):
        self.eval()
        for n in range(0, len(texts), batch_size):
            batch = texts[n:n+batch_size]
            xs = [
                self.vocabulary.sent_to_numerical(
                    self.vocabulary.mode.prepare_input(
                        self.vocabulary.prepare(s)
                    )
                )
                for s in batch
            ]
            logging.info("Dealing with batch %s " % (int(n/batch_size)+1))
            tensor, sentence_length, order = self.vocabulary.pad_and_tensorize(
                    [x for x, _ in xs],
                    padding=max(list(map(lambda x: x[1], xs)))
                )

            if device != "cpu":
                tensor, sentence_length = tensor.to(device), sentence_length.to(device)

            translations = self._string_predict(
                tensor, sentence_length,
                override_src=[batch[order_id] for order_id in order]
            )
            for index in range(len(translations)):
                yield "".join(translations[order.index(index)])

    @staticmethod
    def _apply_max_size(tokens: str, size: int):
        # Use finditer when applied to things with spaces ?
        #  [(m.start(0), m.end(0)) for m in re.finditer(pattern, string)] ?
        current = []
        for tok in re.split(r"(\s+)", tokens):
            if not tok:
                continue
            current.append(tok)
            string_size = len("".join(current))
            if string_size > size:
                yield "".join(current[:-1])
                current = current[-1:]
            elif string_size == size:
                yield "".join(current)
                current = []
        if current:
            yield "".join(current)

    def annotate_text(self, single_sentence, splitter: Optional[str] = None, batch_size=32, device: str = "cpu", rolling=True):
        if splitter is None:
            # ToDo: Mode specific splitter ?
            splitter = r"([\.!\?]+)"

        splitter = re.compile(splitter)
        sentences = [tok for tok in splitter.split(single_sentence) if tok.strip()]
                
        if self._maximum_sentence_size:
            # This is currently quite limitating.
            # If the end token is ending with a W and not a WB, there is no way to "correct it"
            # We'd need a rolling system: cut in the middle of maximum sentence size ?
            treated = []
            max_size = self._maximum_sentence_size
            for single_sentence in sentences:
                if len(single_sentence) > max_size:
                    treated.extend(self._apply_max_size(single_sentence, max_size))
                else:
                    treated.append(single_sentence)
            sentences = treated

        yield from self.annotate(sentences, batch_size=batch_size, device=device)

    @classmethod
    def load(cls, fpath="./model.boudams_model", device=None):
        with tarfile.open(utils.ensure_ext(fpath, 'boudams_model'), 'r') as tar:
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))

            vocab = LabelEncoder.load(
                json.loads(utils.get_gzip_from_tar(tar, "vocabulary.json"))
            )

            obj = cls(vocabulary=vocab, **settings)

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
                # Strict false for predict (nll_weight is removed)
                obj.load_state_dict(torch.load(dictpath,  map_location=device), strict=False)

        obj.eval()

        return obj

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
                torch.save(self.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

        return fpath

    def _string_predict(
            self,
            src,
            src_len,
            override_src: Optional[List[str]] = None
    ) -> torch.Tensor:
        """ Predicts value for a given tensor
        :param src: tensor(batch size x sentence_length)
        :param src_len: tensor(batch size)
        :param label_encoder: Encoder
        :return: Reversed Batch
        """
        out = self(src, src_len)
        logits = torch.argmax(out, -1)
        return self.vocabulary.reverse_batch(
            input_batch=src,
            mask_batch=logits,
            override_numerical_input=override_src
        )
