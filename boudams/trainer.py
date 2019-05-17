import os
import random
import json
import math
import tarfile
import uuid
import enum
import statistics

from collections import namedtuple

import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import tqdm

from torchtext.data import ReversibleField, BucketIterator
from sklearn.metrics import precision_score, recall_score, accuracy_score


from boudams.dataset import Dataset
from boudams.tagger import Seq2SeqTokenizer, DEVICE
from boudams.encoder import DatasetIterator
import boudams.utils as utils


INVALID = "<INVALID>"
Score = namedtuple("Score", ["loss", "perplexity", "accuracy"])


class PlateauModes(enum.Enum):
    loss = "min"
    accuracy = "max"


class EarlyStopException(Exception):
    """ Exception thrown when things plateau """


class Scorer(object):
    """
    Accumulate predictions over batches and compute evaluation scores
    """
    def __init__(self, tagger: Seq2SeqTokenizer):
        self.hypotheses = []
        self.targets = []
        self.tagger: Seq2SeqTokenizer = tagger
        self.tokens = []  # Should be trues as tokens
        self.accuracies = []

    def get_accuracy(self) -> float:
        return statistics.mean(self.accuracies)

    def register_batch(self, hypotheses, targets, verbose: bool = True):
        """
        hyps : list
        targets : list
        tokens : list
        """
        # Makes numbers become STRINGS !
        out, exp = hypotheses.t(), targets.t()

        with torch.cuda.device_of(out):
            out = out.tolist()
        with torch.cuda.device_of(exp):
            exp = exp.tolist()

        for y_true, y_pred in zip(exp, out):
            self.accuracies.append(accuracy_score(y_true[1:], y_pred))

        if verbose:
            show = random.randint(0, len(out)-1)


class LRScheduler(object):
    def __init__(self, optimizer, mode=PlateauModes.loss, **kwargs):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode.value, **kwargs)  # Max because accuracy :)
        self.mode = mode

    def step(self, score):
        self.lr_scheduler.step(getattr(score, self.mode.name))

    @property
    def steps(self):
        return self.lr_scheduler.num_bad_epochs

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
    def __init__(self, tagger: Seq2SeqTokenizer, device: str = DEVICE):
        self.tagger = tagger
        self.device = device

    def _temp_save(self, file_path: str, best_score: float, current_score: Score) -> float:
        if current_score.loss != float("inf") and current_score.loss < best_score:
            torch.save(self.tagger.model.state_dict(), file_path)
            best_score = current_score.loss
        return best_score

    def run(
            self, train_dataset: DatasetIterator, dev_dataset: DatasetIterator,
            lr: float = 1e-3, min_lr: float = 1e-6, lr_factor: int = 0.75, lr_patience: float = 10,
            lr_grace_periode: int = 10,  # Number of first iterations where we ignore lr_patience
            n_epochs: int = 10, batch_size: int = 256, clip: int = 1,
            _seed: int = 1234, fpath: str = "model.tar",
            mode="accuracy",
            after_epoch_fn=None
    ):
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
        dev_score = Score(float("inf"), float("inf"), 0)  # In case exception was run before eval

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
                        str(train_score.loss), str(train_score.perplexity), str(train_score.accuracy),
                        str(dev_score.loss), str(dev_score.perplexity), str(dev_score.accuracy),
                        "UNK", "UNK"
                    )
                )

                # Run a check on saving the current model
                self._temp_save(fid, best_valid_loss, dev_score)

                # Advance Learning Rate if needed
                lr_scheduler.step(dev_score)

                print(f'\tTrain Loss: {train_score.loss:.3f} | Perplexity: {train_score.perplexity:7.3f}'
                      f' Acc.: {train_score.accuracy:.3f}')
                print(f'\t Val. Loss: {dev_score.loss:.3f} | Perplexity: {dev_score.perplexity:7.3f} |'
                      f' Acc.: {dev_score.accuracy:.3f}')
                print(lr_scheduler)
                print()

                if lr_scheduler.steps >= lr_patience and lr_scheduler.lr < min_lr:
                    raise EarlyStopException()

                if epoch == lr_grace_periode:
                    lr_scheduler.lr_scheduler.patience = lr_patience

            except KeyboardInterrupt:
                print("Interrupting training...")
                break
            except EarlyStopException:
                print("Reached plateau for too long, stopping.")

        self._temp_save(fid, best_valid_loss, dev_score)
        try:
            self.tagger.model.load_state_dict(torch.load(fid))
            os.remove(fid)
        except FileNotFoundError:
            print("No model was saved during training")

        self.save(fpath, csv_content)

        print("Saved !")
        if after_epoch_fn:
            after_epoch_fn(self)

    def save(self, fpath="model.tar", csv_content=None):

        fpath = utils.ensure_ext(fpath, 'tar', infix=None)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if not os.path.isdir(dirname):
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
    def init_csv_content():
        return [
            (
                "Epoch",
                "Train Loss", "Train Perplexity", "Train Accuracy",
                "Dev Loss", "Dev Perplexity", "Dev Accuracy",
                "Test Loss", "Test Perplexity"
            )
        ]

    def _get_perplexity(self, loss):
        try:
            return math.exp(loss)
        except:
            return float("inf")

    def _train_epoch(self, iterator: DatasetIterator, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss,
                     clip: float, desc: str, batch_size: int = 32) -> Score:
        self.tagger.model.train()

        epoch_loss = 0

        scorer = Scorer(self.tagger)

        batch_generator = iterator.get_epoch(batch_size=batch_size, device=self.device)
        batches = batch_generator()

        for _ in tqdm.tqdm(range(0, iterator.batch_count), desc=desc):
            src, src_len, trg, trg_len = next(batches)

            optimizer.zero_grad()

            src_in, trg_in = self.tagger.model._reshape_input(src, trg)

            output, attention = self.tagger.model(src_in, src_len, trg_in)

            scorer.register_batch(
                self.tagger.model._reshape_output_for_scorer(output),
                trg
            )

            # We redim to work like other models
            output_loss, trg_loss = self.tagger.model._reshape_out_for_loss(output, trg)

            # output = [batch size * trg sent len - 1, output dim]
            # trg = [batch size * trg sent len - 1]

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output_loss, trg_loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.tagger.model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        loss = epoch_loss / len(iterator)
        return Score(loss, self._get_perplexity(loss), scorer.get_accuracy())

    def evaluate(self, iterator: DatasetIterator, criterion: nn.CrossEntropyLoss,
                 desc: str, batch_size: int) -> Score:

        self.tagger.model.eval()

        epoch_loss = 0

        scorer = Scorer(self.tagger)

        with torch.no_grad():
            batch_generator = iterator.get_epoch(batch_size=batch_size, device=self.device)
            batches = batch_generator()

            for _ in tqdm.tqdm(range(0, iterator.batch_count), desc=desc):
                src, src_len, trg, trg_len = next(batches)

                src_in, trg_in = self.tagger.model._reshape_input(src, trg)

                output, attention = self.tagger.model(
                    src_in, src_len, trg_in, teacher_forcing_ratio=0
                )  # turn off teacher forcing

                # We register the current batch
                #  For this to work, we get ONLY the best score of output which mean we need to argmax
                #   at the second layer (base 0 I believe)
                # We basically get the best match at the output dim layer : the best character.
                scorer.register_batch(
                    self.tagger.model._reshape_output_for_scorer(output),
                    trg
                )

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]
                output_loss, trg_loss = self.tagger.model._reshape_out_for_loss(output, trg)

                # trg = [(trg sent len - 1) * batch size]
                # output = [(trg sent len - 1) * batch size, output dim]
                loss = criterion(output_loss, trg_loss)
                epoch_loss += loss.item()

        loss = epoch_loss / len(iterator)

        return Score(loss, self._get_perplexity(loss), scorer.get_accuracy())

    def test(self, test_dataset: Dataset, batch_size: int = 256):
        test_iterator = BucketIterator.splits(
            datasets=[test_dataset],
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device
        )

        # Set up loss but ignore the loss when the token is <pad>
        #     where <pad> is the token for filling the vector to get same-sized matrix
        PAD_IDX = self.tagger.vocabulary.vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        test_loss = self.evaluate(test_iterator, criterion, desc="Test")

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {test_loss.perplexity:7.3f} |'
              f'Test Accuracy {test_loss.accuracy}')
