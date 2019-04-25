import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import dill

from torchtext.data import ReversibleField, BucketIterator

import random
import time
import os
import math
import json
import tqdm
import tarfile
from typing import List, Tuple

from .model import gru, lstm, bidir
from .dataset import build_vocab, CharacterField, TabularDataset as Dataset, get_datasets, InputDataset, SOS_TOKEN
from . import utils


teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


class Seq2SeqTokenizer:
    def __init__(
            self,
            vocabulary: ReversibleField,
            hidden_size: int = 256, n_layers: int = 2, emb_enc_dim: int = 256, emb_dec_dim: int = 256,
            enc_hid_dim: int = None, dec_hid_dim: int = None,
            enc_dropout: float= 0.5, dec_dropout: float = 0.5,
            device: str = DEVICE, system: str = "bi-gru"
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

        self.vocabulary: ReversibleField = vocabulary
        self.vocabulary_dimension: int = len(self.vocabulary.vocab)

        self.device: str = device
        self.n_layers: int = n_layers
        self.enc_hid_dim = self.dec_hid_dim = self.hidden_size = hidden_size
        if enc_hid_dim and dec_hid_dim:
            self.enc_hid_dim: int = enc_hid_dim
            self.dec_hid_dim: int = dec_hid_dim

        self.emb_enc_dim: int = emb_enc_dim
        self.emb_dec_dim: int = emb_dec_dim
        self.enc_dropout: float = enc_dropout
        self.dec_dropout: float = dec_dropout
        self.system: str = system

        if self.system == "gru":
            self.enc: gru.Encoder = gru.Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size,
                                                self.enc_dropout)
            self.dec: gru.Decoder = gru.Decoder(self.vocabulary_dimension, self.emb_dec_dim, self.hidden_size,
                                                self.dec_dropout)

            self.model: gru.Seq2Seq = gru.Seq2Seq(self.enc, self.dec, self.device).to(device)
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
                self.vocabulary_dimension, emb_dim=self.emb_dec_dim,
                enc_hid_dim=self.enc_hid_dim, dec_hid_dim=self.dec_hid_dim, dropout=self.enc_dropout,
                attention=self.attention
            )
            self.init_weights = bidir.init_weights
            self.model: gru.Seq2Seq = bidir.Seq2Seq(self.enc, self.dec, self.device).to(device)

        else:
            self.enc: lstm.Encoder = lstm.Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size,
                                                  self.n_layers, self.enc_dropout)
            self.dec: lstm.Decoder = lstm.Decoder(self.vocabulary_dimension, self.emb_dec_dim, self.hidden_size,
                                                  self.n_layers, self.dec_dropout)
            self.init_weights = lstm.init_weights
            self.model: lstm.Seq2Seq = lstm.Seq2Seq(self.enc, self.dec, self.device).to(device)

        self._dataset = None

    @staticmethod
    def get_dataset_and_vocabularies(
            train, dev, test
    ) -> Tuple[ReversibleField, Dataset, Dataset, Dataset]:
        """

        :param train: Path to train TSV file
        :param dev:  Path to dev TSV file
        :param test:  Path to test TSV file
        :return:
        """
        train, dev, test = get_datasets(train, dev, test)
        vocab = build_vocab(CharacterField, (train, dev, test))
        return vocab, train, dev, test

    @property
    def padtoken(self):
        return self.vocabulary.vocab.stoi[self.vocabulary.pad_token]

    def train(
            self, train_dataset: Dataset, dev_dataset: Dataset,
            n_epochs: int = 10, batch_size: int = 256, clip: int = 1,
            _seed: int = 1234,
            fpath: str = "model.tar"
    ):
        """

        :param train_dataset:
        :param dev_dataset:
        :param n_epochs:
        :param batch_size:
        :param clip:
        :param _seed:
        :return:
        """
        random.seed(_seed)
        torch.manual_seed(_seed)
        torch.backends.cudnn.deterministic = True

        self.model.apply(self.init_weights)

        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters())

        # Set up loss but ignore the loss when the token is <pad>
        #     where <pad> is the token for filling the vector to get same-sized matrix
        criterion = nn.CrossEntropyLoss(ignore_index=self.padtoken)

        # Set-up the iterators
        train_iterator, dev_iterator = BucketIterator.splits(
            datasets=(train_dataset, dev_dataset),
            device=self.device,
            batch_size=batch_size,
            sort=False  # https://github.com/fastai/fastai/pull/183#issuecomment-477134781
        )

        best_valid_loss = float('inf')

        for epoch in range(1, n_epochs+1):
            print("[Epoch %s/%s]" % (epoch, n_epochs), flush=True)
            try:
                start_time = time.time()

                train_loss = self._train_epoch(train_iterator, optimizer, criterion, clip)
                valid_loss = self.evaluate(dev_iterator, criterion)

                end_time = time.time()

                epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save(fpath)

                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            except KeyboardInterrupt:
                print("Interrupting training...")
                self.save(fpath)
                print("Saved !")
                break

    def _train_epoch(self, iterator: BucketIterator, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss,
                     clip: float):
        """

        :param iterator:
        :param optimizer:
        :param criterion:
        :param clip: Cliping
        :return:
        """
        self.model.train()

        epoch_loss = 0

        for i, batch in enumerate(tqdm.tqdm(iterator)):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output = self.model(src, trg)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, iterator: BucketIterator, criterion: nn.CrossEntropyLoss):

        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = self.model(src, trg, teacher_forcing_ratio=0)  # turn off teacher forcing

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]

                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

                # trg = [(trg sent len - 1) * batch size]
                # output = [(trg sent len - 1) * batch size, output dim]

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def tag(self, iterator: BucketIterator):

        for i, batch in enumerate(iterator):
            src = batch.src
            output = self.model(
                src, trg=None,
                teacher_forcing_ratio=0,
                sos_token=lambda device: SOS_TOKEN(device=self.device, field=self.vocabulary)
            )  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [Maximum Sentence Length, Number of Sentence in batch, Number of possible characters]
            _, ind = torch.topk(output, 1, dim=2)
            # ind = [Maximum Sentence Length, Number of Sentences in Batch, One Result]

            # output = output[1:].view(-1, output.shape[-1])

            yield ind.squeeze().permute(1, 0)

    def test(self, test_dataset: Dataset, batch_size: int = 256):
        test_iterator = BucketIterator.splits(
            datasets=[test_dataset],
            device=self.device,
            batch_size=batch_size,
            sort=False  # https://github.com/fastai/fastai/pull/183#issuecomment-477134781
        )

        # Set up loss but ignore the loss when the token is <pad>
        #     where <pad> is the token for filling the vector to get same-sized matrix
        PAD_IDX = self.vocabulary.vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        test_loss = self.evaluate(test_iterator, criterion)

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    @property
    def settings(self):
        return {
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "enc_hid_dim": self.enc_hid_dim,
            "dec_hid_dim": self.dec_hid_dim,
            "emb_enc_dim": self.emb_enc_dim,
            "emb_dec_dim": self.emb_dec_dim,
            "enc_dropout": self.enc_dropout,
            "dec_dropout": self.dec_dropout,
            "system": self.system
        }

    def save(self, fpath="model.tar"):

        fpath = utils.ensure_ext(fpath, 'tar', infix=None)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(fpath, 'w') as tar:

            # serialize settings
            string, path = json.dumps(self.settings), 'settings.json.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize field
            with utils.tmpfile() as tmppath:
                torch.save(self.model.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

            # serialize field
            with utils.tmpfile() as tmppath:
                torch.save(self.vocabulary, tmppath, pickle_module=dill)
                tar.add(tmppath, arcname='vocabulary.pt')

        return fpath

    @classmethod
    def load(cls, fpath="./model.tar"):
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('vocabulary.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'vocabulary.pt')
                vocab = torch.load(dictpath, pickle_module=dill)

            obj = cls(vocabulary=vocab, **settings)

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
                obj.model.load_state_dict(torch.load(dictpath))

        obj.model.eval()

        return obj

    def annotate(self, texts: List[str]):
        data = InputDataset(texts, vocabulary=self.vocabulary)
        old_batch_first = self.vocabulary.batch_first
        self.vocabulary.batch_first = True
        yield from [
            line
            for batch_out in self.tag(data.get_iterator())
            for line in self.vocabulary.reverse(batch_out)
        ]
        self.vocabulary.batch_first = old_batch_first

