import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import dill

from torchtext.data import Field, BucketIterator

import random
import time
import math
from typing import List, Tuple

from .model import Encoder, Decoder, Seq2Seq
from .dataset import build_vocab, CharacterField, TabularDataset as Dataset, get_datasets
from .utils import epoch_time


teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MAX_LENGTH = 150


class Seq2SeqTokenizer:
    def __init__(
            self,
            vocabulary,
            hidden_size: int = 256, n_layers: int = 2, emb_enc_dim: int= 256, emb_dec_dim: int = 256,
            enc_dropout: float= 0.5, dec_dropout: float= 0.5,
             max_length=MAX_LENGTH, device: str=DEVICE
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

        self.vocabulary: Field = vocabulary
        self.vocabulary_dimension: int = len(self.vocabulary.vocab)

        self.device: str = device
        self.n_layers: int = n_layers
        self.hidden_size: int = hidden_size

        self.emb_enc_dim: int = emb_enc_dim
        self.emb_dec_dim: int = emb_dec_dim
        self.enc_dropout: float = enc_dropout
        self.dec_dropout: float = dec_dropout

        self.enc: Encoder = Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size, self.n_layers,
                                    self.enc_dropout)
        self.dec: Decoder = Decoder(self.vocabulary_dimension, self.emb_dec_dim, self.hidden_size, self.n_layers,
                                    self.dec_dropout)

        self.model: Seq2Seq = Seq2Seq(
            self.enc,
            self.dec,
            self.device
        ).to(device)

        self._dataset = None

    @staticmethod
    def get_dataset_and_vocabularies(
            train, dev, test
    ) -> Tuple[Field, Dataset, Dataset, Dataset]:
        """

        :param train: Path to train TSV file
        :param dev:  Path to dev TSV file
        :param test:  Path to test TSV file
        :return:
        """
        train, dev, test = get_datasets(train, dev, test)
        vocab = build_vocab(CharacterField, (train, dev, test))
        return vocab, train, dev, test

    def train(
            self, train_dataset: Dataset, dev_dataset: Dataset,
            n_epochs: int = 10, batch_size: int = 256, clip: int = 1,
            _seed: int = 1234
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

        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters())

        # Set up loss but ignore the loss when the token is <pad>
        #     where <pad> is the token for filling the vector to get same-sized matrix
        PAD_IDX = self.vocabulary.vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        # Set-up the iterators
        train_iterator, dev_iterator = BucketIterator.splits(
            datasets=(train_dataset, dev_dataset),
            device=self.device,
            batch_size=batch_size,
            sort=False  # https://github.com/fastai/fastai/pull/183#issuecomment-477134781
        )

        best_valid_loss = float('inf')

        for epoch in range(n_epochs):

            try:
                start_time = time.time()

                train_loss = self._train_epoch(train_iterator, optimizer, criterion, clip)
                valid_loss = self.evaluate(dev_iterator, criterion)

                end_time = time.time()

                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save()

                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            except KeyboardInterrupt:
                self.save()
                self.save_vocab()

        self.save_vocab()

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

        for i, batch in enumerate(iterator):
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
            output = self.model(src, None, 0)  # turn off teacher forcing

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

    def save_vocab(self):
        torch.save(self.vocabulary, "vocabulary.pt", pickle_module=dill)
        #with open("./vocabulary.pickle", "wb") as f:
        #    pickle.dump(self.vocabulary, f)

    def save(self):
        torch.save(self.model.state_dict(), "./seq2seq.pt")

    def evaluateRandomly(self, pairs, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
