import random
import json
import tarfile
import uuid
import enum

from collections import namedtuple
from typing import Callable, List, Tuple


import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import tqdm

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

from boudams.tagger import BoudamsTagger, DEVICE
from boudams.encoder import DatasetIterator
import boudams.utils as utils
import os

INVALID = "<INVALID>"
DEBUG = bool(os.getenv("DEBUG"))
Score = namedtuple("Score", ["loss", "accuracy", "precision", "recall", "fscore", "scorer"])


class PlateauModes(enum.Enum):
    loss = "min"
    accuracy = "max"
    leven = "min"
    leven_per_char = "min"


class EarlyStopException(Exception):
    """ Exception thrown when things plateau """


class Scorer(object):
    """
    Accumulate predictions over batches and compute evaluation scores
    """
    def __init__(self, tagger: BoudamsTagger, masked: bool = False, record: bool = False):
        """

        :param tagger: Tagger
        :param masked: Mask with Word Boundary
        :param record: Record all inputs
        """
        self.hypotheses = []
        self.targets = []
        self.tagger: BoudamsTagger = tagger
        self.tokens = []  # Should be trues as tokens
        self.trues = []
        self.preds = []
        self.srcs = []
        self.report = ""

        self._score_tuple = namedtuple("scores", ["accuracy", "precision", "recall", "fscore"])
        self.scores = None
        self.masked: bool = masked

    def plot_confusion_matrix(self, path: str = "confusion-matrix.png"):
        try:
            from .utils import plot_confusion_matrix, plt
        except ImportError:
            print("You need to install matplotlib for this feature")
            raise
        unrolled_trues = list([y_char for y_sent in self.trues for y_char in y_sent])
        unrolled_preds = list([y_char for y_sent in self.preds for y_char in y_sent])

        plot_confusion_matrix(
            unrolled_trues,
            unrolled_preds,
            labels=[self.tagger.vocabulary.space_token_index, self.tagger.vocabulary.mask_token_index],
            classes=["WordBoundary", "WordContent"]
            if self.tagger.vocabulary.space_token_index < self.tagger.vocabulary.mask_token_index else
            ["WordContent", "WordBoundary"]
        )
        plt.savefig(path)

    def compute(self, class_report=False) -> "Scorer":
        unrolled_trues = list([y_char for y_sent in self.trues for y_char in y_sent])
        unrolled_preds = list([y_char for y_sent in self.preds for y_char in y_sent])

        matrix = confusion_matrix(
            unrolled_trues,
            unrolled_preds,
            labels=[self.tagger.vocabulary.space_token_index, self.tagger.vocabulary.mask_token_index]
        )
        # Accuracy score takes into account PAD, EOS and SOS so we get the data from the confusion matrix
        samples = sum(sum(matrix))
        errors = matrix[0][1] + matrix[1][0]
        accuracy = 1 - (errors / samples)

        # Technically, data is padded, so we can unroll things
        precision, recall, fscore, _ = precision_recall_fscore_support(
            unrolled_trues, unrolled_preds, average="macro",
            # Ignore pad errors
            labels=[self.tagger.vocabulary.space_token_index, self.tagger.vocabulary.mask_token_index]
        )

        if class_report:
            self.report = classification_report(
                y_pred=unrolled_preds,
                y_true=unrolled_trues,
                labels=[self.tagger.vocabulary.space_token_index, self.tagger.vocabulary.mask_token_index],
                target_names=["Word Boundary", "Word Character"]
            )

        self.scores = self._score_tuple(accuracy, precision, recall, fscore)

        return self

    def get_accuracy(self) -> float:
        if not self.scores:
            self.compute()
        return self.scores.accuracy

    def register_batch(self, hypotheses, targets, src):
        """

        :param hypotheses: tensor(batch size x sentence length)
        :param targets: tensor(batch size x sentence length)
        """
        with torch.cuda.device_of(hypotheses):
            out = hypotheses.tolist()
        with torch.cuda.device_of(targets):
            exp = targets.tolist()
        with torch.cuda.device_of(src):
                src = src.tolist()

        for y_true, y_pred, x in zip(exp, out, src):
            stop = x.index(self.tagger.padtoken) if self.tagger.padtoken in x else len(x)
            self.trues.append(y_true[:stop])
            self.preds.append(y_pred[:stop])
            self.srcs.append(x[:stop])


class LRScheduler(object):
    def __init__(self, optimizer, mode=PlateauModes.accuracy, **kwargs):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode.value, **kwargs)  # Max because accuracy :)
        self.mode = mode
        self.steps = 0

    def step(self, score):
        scheduler_steps = self.lr_scheduler.num_bad_epochs
        self.lr_scheduler.step(getattr(score, self.mode.name))
        # No change in number of bad epochs =
        #   we are progressing
        if scheduler_steps == self.lr_scheduler.num_bad_epochs:
            self.steps = 0
        # Otherwise, we are not
        else:
            self.steps += 1

        if self.steps >= self.patience * 2:
            # If we haven't progressed even by lowering twice
            raise EarlyStopException("No progress for %s , stoping now... " % self.steps)

    @property
    def patience(self):
        return self.lr_scheduler.patience

    @property
    def lr(self):
        return self.lr_scheduler.optimizer.param_groups[0]['lr']

    def __repr__(self):
        return '<LrScheduler lr="{}" lr_steps="{}" lr_patience="{}"/>' \
            .format(self.lr_scheduler.optimizer.param_groups[0]['lr'],
                    self.lr_scheduler.num_bad_epochs,
                    self.lr_scheduler.patience)


class Trainer(object):
    def __init__(self, tagger: BoudamsTagger, device: str = DEVICE):
        self.tagger = tagger
        self.device = device
        self.debug = False

    def _temp_save(self, file_path: str, best_score: float, current_score: Score) -> float:
        if current_score.loss != float("inf") and current_score.loss < best_score:
            torch.save(self.tagger.model.state_dict(), file_path)
            best_score = current_score.loss
        return best_score

    @staticmethod
    def print_score(key: str, score: Score) -> None:
        print(f'\t{key} Loss: {score.loss:.3f} | FScore: {score.fscore:.3f} | '
              f' Acc.: {score.accuracy:.3f} | '
              f' Prec.: {score.precision:.3f} | '
              f' Recl.: {score.recall:.3f}')

    def run(
            self, train_dataset: DatasetIterator, dev_dataset: DatasetIterator,
            lr: float = 1e-3, min_lr: float = 1e-6, lr_factor: int = 0.75, lr_patience: float = 10,
            lr_grace_periode: int = 10,  # Number of first iterations where we ignore lr_patience
            n_epochs: int = 10, batch_size: int = 256, clip: int = 1,
            _seed: int = 1234, fpath: str = "model.tar",
            mode="loss",
            debug: Callable[[BoudamsTagger], None] = None
    ):
        if _seed:
            random.seed(_seed)
            torch.manual_seed(_seed)
            torch.backends.cudnn.deterministic = True

        if self.tagger.init_weights is not None:
            self.tagger.model.apply(self.tagger.init_weights)

        # Set up optimizer
        optimizer = optim.Adam(self.tagger.model.parameters(), lr=lr)

        # Set-up LR Scheduler
        lr_scheduler = LRScheduler(
            optimizer,
            factor=lr_factor, patience=lr_grace_periode, min_lr=min_lr,
            mode=getattr(PlateauModes, mode)
        )

        # Generates a temp file to store the best model
        fid = '/tmp/{}'.format(str(uuid.uuid1()))
        best_valid_loss = float("inf")
        # In case exception was run before eval
        dev_score = Score(float("inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), None)

        # Set up loss but ignore the loss when the token is <pad>
        #     where <pad> is the token for filling the vector to get same-sized matrix
        criterion = nn.CrossEntropyLoss(ignore_index=self.tagger.padtoken)

        csv_content = self.init_csv_content()
        for epoch in range(1, n_epochs+1):
            try:
                train_score = self._train_epoch(
                    train_dataset, optimizer, criterion, clip,
                    desc="[Epoch Training %s/%s]" % (epoch, n_epochs),
                    batch_size=batch_size
                )
                dev_score = self.evaluate(
                    dev_dataset, criterion,
                    desc="[Epoch Dev %s/%s]" % (epoch, n_epochs),
                    batch_size=batch_size
                )

                # Get some CSV content
                csv_content.append(
                    (
                        str(epoch),
                        # train
                        str(train_score.loss), str(train_score.accuracy), str(train_score.precision),
                            str(train_score.recall), str(train_score.fscore),
                        # Dev
                        str(dev_score.loss), str(dev_score.accuracy), str(dev_score.precision),
                            str(dev_score.recall), str(dev_score.fscore),
                        # Test
                        "UNK", "UNK", "UNK", "UNK", "UNK"
                    )
                )

                # Run a check on saving the current model
                best_valid_loss = self._temp_save(fid, best_valid_loss, dev_score)
                self.print_score("Train", train_score)
                self.print_score("Dev", dev_score)
                print(lr_scheduler)
                print()

                # Advance Learning Rate if needed
                lr_scheduler.step(dev_score)

                if lr_scheduler.steps >= lr_patience and lr_scheduler.lr < min_lr:
                    raise EarlyStopException()

                if epoch == lr_grace_periode:
                    lr_scheduler.lr_scheduler.patience = lr_patience

                if debug is not None:
                    debug(self.tagger)

            except KeyboardInterrupt:
                print("Interrupting training...")
                break
            except EarlyStopException:
                print("Reached plateau for too long, stopping.")
                break

        best_valid_loss = self._temp_save(fid, best_valid_loss, dev_score)

        try:
            self.tagger.model.load_state_dict(torch.load(fid))
            print("Saving model with loss %s " % best_valid_loss)
            os.remove(fid)
        except FileNotFoundError:
            print("No model was saved during training")

        self.save(fpath, csv_content)

        print("Saved !")

    def save(self, fpath="model.tar", csv_content=None):

        fpath = utils.ensure_ext(fpath, 'tar', infix=None)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        if csv_content:
            with open(fpath.replace(".tar", ".csv"), "w") as f:
                for line in csv_content:
                    f.write(";".join([str(x) for x in line])+"\n")

        with tarfile.open(fpath, 'w') as tar:

            # serialize settings
            string, path = json.dumps(self.tagger.settings), 'settings.json.zip'
            utils.add_gzip_to_tar(string, path, tar)

            string, path = self.tagger.vocabulary.dump(), 'vocabulary.json'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize field
            with utils.tmpfile() as tmppath:
                torch.save(self.tagger.model.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

        return fpath

    @staticmethod
    def init_csv_content() -> List[Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str]]:
        return [
            (
                "Epoch",
                "Train Loss", "Train Accuracy", "Train Precision", "Train Recall", "Train F1",
                "Dev Loss", "Dev Accuracy", "Dev Precision", "Dev Recall", "Dev F1",
                "Test Loss", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"
            )
        ]

    def _train_epoch(self, iterator: DatasetIterator, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss,
                     clip: float, desc: str, batch_size: int = 32) -> Score:
        self.tagger.model.train()

        epoch_loss = 0

        scorer = Scorer(self.tagger)

        batch_generator = iterator.get_epoch(
            batch_size=batch_size,
            device=self.device
        )
        batches = batch_generator()

        for batch_index in tqdm.tqdm(range(0, iterator.batch_count), desc=desc):
            src, src_len, trg, _ = next(batches)

            optimizer.zero_grad()

            loss = self.tagger.model.gradient(
                src, src_len, trg,
                scorer=scorer, criterion=criterion
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.tagger.model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        loss = epoch_loss / iterator.batch_count
        scorer.compute()
        return Score(loss,
                     accuracy=scorer.scores.accuracy,
                     precision=scorer.scores.precision,
                     recall=scorer.scores.recall,
                     fscore=scorer.scores.fscore,
                     scorer=scorer)

    def evaluate(self, iterator: DatasetIterator, criterion: nn.CrossEntropyLoss,
                 desc: str, batch_size: int, test_mode=False,
                 class_report: bool = False) -> Score:

        self.tagger.model.eval()

        epoch_loss = 0

        scorer = Scorer(self.tagger, record=test_mode is True)

        with torch.no_grad():
            batch_generator = iterator.get_epoch(
                batch_size=batch_size,
                device=self.device
            )
            batches = batch_generator()

            for _ in tqdm.tqdm(range(0, iterator.batch_count), desc=desc):
                src, src_len, trg, _ = next(batches)

                loss = self.tagger.model.gradient(
                    src, src_len, trg,
                    scorer=scorer, criterion=criterion,
                    evaluate=True
                )
                epoch_loss += loss.item()

        loss = epoch_loss / iterator.batch_count

        scorer.compute(class_report=class_report)
        return Score(loss,
                     accuracy=scorer.scores.accuracy,
                     precision=scorer.scores.precision,
                     recall=scorer.scores.recall,
                     fscore=scorer.scores.fscore,
                     scorer=scorer)

    def test(self, test_dataset: DatasetIterator, batch_size: int = 256, do_print=True, class_report=False):
        # Set up loss but ignore the loss when the token is <pad>
        #     where <pad> is the token for filling the vector to get same-sized matrix
        criterion = nn.CrossEntropyLoss(ignore_index=self.tagger.vocabulary.pad_token_index)

        score_object = self.evaluate(test_dataset, criterion, desc="Test", batch_size=batch_size, test_mode=True,
                                     class_report=class_report)
        scorer: Scorer = score_object.scorer

        if do_print:
            self.print_score("Test", score_object)
        return scorer
