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
from .dataset import build_vocab, CharacterField, TabularDataset as Dataset, get_datasets, InputDataset
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
            out_max_sentence_length: int = 150,
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
        self.out_max_sentence_length: int = out_max_sentence_length
        self.system: str = system

        seq2seq_shared_params = {
            "pad_idx": self.padtoken,
            "sos_idx": self.sostoken,
            "eos_idx": self.eostoken,
            "device": self.device,
            "out_max_sentence_length": self.out_max_sentence_length
        }

        if self.system == "gru":
            self.enc: gru.Encoder = gru.Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size,
                                                self.enc_dropout)
            self.dec: gru.Decoder = gru.Decoder(self.vocabulary_dimension, self.emb_dec_dim, self.hidden_size,
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
                self.vocabulary_dimension, emb_dim=self.emb_dec_dim,
                enc_hid_dim=self.enc_hid_dim, dec_hid_dim=self.dec_hid_dim, dropout=self.enc_dropout,
                attention=self.attention
            )
            self.init_weights = bidir.init_weights
            self.model: gru.Seq2Seq = bidir.Seq2Seq(self.enc, self.dec, **seq2seq_shared_params).to(device)

        else:
            self.enc: lstm.Encoder = lstm.Encoder(self.vocabulary_dimension, self.emb_enc_dim, self.hidden_size,
                                                  self.n_layers, self.enc_dropout)
            self.dec: lstm.Decoder = lstm.Decoder(self.vocabulary_dimension, self.emb_dec_dim, self.hidden_size,
                                                  self.n_layers, self.dec_dropout)
            self.init_weights = lstm.init_weights
            self.model: lstm.Seq2Seq = lstm.Seq2Seq(self.enc, self.dec, **seq2seq_shared_params).to(device)

        self._dataset = None

    @property
    def padtoken(self):
        return self.vocabulary.vocab.stoi[self.vocabulary.pad_token]

    @property
    def sostoken(self):
        return self.vocabulary.vocab.stoi[self.vocabulary.init_token]

    @property
    def eostoken(self):
        return self.vocabulary.vocab.stoi[self.vocabulary.eos_token]

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

    def init_csv_content(self):
        return [
            ("Epoch", "Train Loss", "Train Perplexity", "Dev Loss", "Dev Perplexity", "Test Loss", "Test Perplexity")
        ]

    def train(
            self, train_dataset: Dataset, dev_dataset: Dataset,
            n_epochs: int = 10, batch_size: int = 256, clip: int = 1,
            _seed: int = 1234, fpath: str = "model.tar",
            after_epoch_fn = None
    ):
        """

        :param train_dataset:
        :param dev_dataset:
        :param n_epochs:
        :param batch_size:
        :param clip:
        :param _seed:
        :param fpath:
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
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device
        )

        best_valid_loss = float('inf')

        csv_content = self.init_csv_content()

        for epoch in range(1, n_epochs+1):
            try:
                train_loss = self._train_epoch(train_iterator, optimizer, criterion, clip,
                                               desc="[Epoch Training %s/%s]" % (epoch, n_epochs))
                valid_loss = self.evaluate(dev_iterator, criterion,
                                               desc="[Epoch Dev %s/%s]" % (epoch, n_epochs))

                csv_content.append(
                    (str(epoch), train_loss, math.exp(train_loss), valid_loss, math.exp(valid_loss), "UNK", "UNK")
                )

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save(fpath)

                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
                # Do not work, crashes the batch size for some fucking reason
                #if after_epoch_fn:
                #    after_epoch_fn(self)

            except KeyboardInterrupt:
                print("Interrupting training...")
                break

        self.save(fpath, csv_content)
        print("Saved !")
        if after_epoch_fn:
            after_epoch_fn(self)

    def _train_epoch(self, iterator: BucketIterator, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss,
                     clip: float, desc: str):
        """

        :param iterator:
        :param optimizer:
        :param criterion:
        :param clip: Cliping
        :return:
        """
        self.model.train()

        epoch_loss = 0

        for i, batch in enumerate(tqdm.tqdm(iterator, desc=desc)):
            src, src_len = batch.src
            trg, _ = batch.trg  # We don't care about target length !

            optimizer.zero_grad()
            output, attention = self.model(src, src_len, trg)

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

    def evaluate(self, iterator: BucketIterator, criterion: nn.CrossEntropyLoss, desc: str):

        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(iterator), desc=desc):
                src, src_len = batch.src
                trg, _ = batch.trg  # Length not used

                output, attention = self.model(src, src_len, trg, teacher_forcing_ratio=0)  # turn off teacher forcing

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
        PAD_IDX = self.vocabulary.vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        test_loss = self.evaluate(test_iterator, criterion, desc="Test")

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
            "out_max_sentence_length": self.out_max_sentence_length,
            "system": self.system
        }

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

    def reverse(self, batch):
        if not self.vocabulary.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        ignore = (self.eostoken, self.sostoken, self.padtoken, self.vocabulary.unk_token)
        batch = [
            [self.vocabulary.vocab.itos[ind] for ind in ex[1:] if ind not in ignore]
            for ex in batch
        ]  # denumericalize

        return [''.join(ex) for ex in batch]

    def annotate(self, texts: List[str]):

        self.model.eval()
        for sentence in texts:
            numericalized = [self.vocabulary.vocab.stoi[t] for t in sentence] + [self.eostoken]
            sentence_length = torch.LongTensor([len(numericalized)]).to(self.device)
            tensor = torch.LongTensor(numericalized).unsqueeze(1).to(self.device)
            translation_tensor_logits, attention = self.model(
                tensor, sentence_length, trg=None, teacher_forcing_ratio=0)
            translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
            translation = [self.vocabulary.vocab.itos[t] for t in translation_tensor]
            translation = translation[1:]
            if attention:
                attention = attention[1:]
            yield "".join(translation)#, attention

