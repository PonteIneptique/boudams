import torch
import torch.cuda
import torch.nn
from typing import Tuple, Dict, List, Optional, Iterator
import logging
import collections
import random
import numpy as np
import json


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_INIT_TOKEN = "<SOS>"
DEFAULT_EOS_TOKEN = "<EOS>"
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_UNK_TOKEN = "<UNK>"


class DatasetIterator:
    def __init__(self, label_encoder: "LabelEncoder", file,
                 batch_size: int = 32,
                 batch_first: bool = False, random: bool = False):
        self._l_e = label_encoder

        self.line_starts_offsets: List[int] = []
        self.current_epoch: List[tuple, int] = []
        self.random = random
        self.batch_first = batch_first
        self.file = file
        self.batch_count = 0
        self.batch_size = batch_size

        self._setup()

    def __repr__(self):
        return "<DatasetIterator lines='{}' random='{}' \n" \
               "\tbatch_first='{}' batches='{}' batch_size='{}'/>".format(
                    len(self.line_starts_offsets),
                    self.random,
                    self.batch_first,
                    self.batch_count,
                    self.batch_size
                )


    def _setup(self):
        logging.info("DatasetIterator reading indexes of lines")
        with open(self.file, "rb") as fio:
            offset = 0
            for line in fio:
                if line.strip():  # if line is not empty
                    self.line_starts_offsets.append(offset)
                offset += len(line)

        self.length = len(self.line_starts_offsets)
        logging.info("DatasetIterator found {} lines in {}".format(self.length, self.file))

        # Get the number of batch for TQDM
        self.batch_count = self.length // self.batch_size + bool(self.length % self.batch_size)

    def get_line(self, *line_index):
        with open(self.file, "rb") as fio:
            for line_start in line_index:
                fio.seek(line_start)
                yield self._l_e.readunit(fio.readline().decode("utf-8").strip())

    def get_epoch(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        lines = [] + self.line_starts_offsets
        if self.random:
            random.shuffle(lines)

        for n in range(0, len(lines), self.batch_size):
            xs, y_trues = [], []
            max_len_x, max_len_y = 0, 0  # Needed for padding
            for x, y in self.get_line(*lines[n:self.batch_size]):
                max_len_x = max(len(x), max_len_x)
                max_len_y = max(len(y), max_len_y)
                xs.append(x)
                y_trues.append(y)
            yield self._l_e.tensorize(xs, padding=max_len_x), self._l_e.tensorize(y_trues, padding=max_len_y)


class LabelEncoder:
    def __init__(self, device=DEVICE,
                 init_token=DEFAULT_INIT_TOKEN,
                 eos_token=DEFAULT_EOS_TOKEN,
                 pad_token=DEFAULT_PAD_TOKEN,
                 unk_token=DEFAULT_UNK_TOKEN
                 ):
        self.device = device
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.init_token_index = 0
        self.eos_token_index = 1
        self.pad_token_index = 2
        self.unk_token_index = 3
        self.random = True

        self.itos: Dict[int, str] = {
            self.init_token_index: init_token,
            self.eos_token_index: eos_token,
            self.pad_token_index: pad_token,
            self.unk_token_index: unk_token
        }  # Id to string for reversal

        self.stoi: Dict[str, int] = {
            init_token: self.init_token_index,
            eos_token: self.eos_token_index,
            pad_token: self.pad_token_index,
            unk_token: self.unk_token_index
        }  # String to ID

    def build(self, *paths, debug=False):
        """ Builds vocabulary

        :param paths: Path of file to read
        :return:
        """
        recorded_chars = set()
        counter = None
        if debug:
            counter = collections.Counter()

        logging.info("Reading files for vocabulary building")
        for path in paths:
            with open(path) as fio:
                for line in fio.readlines():

                    x, y_true = self.readunit(line)
                    recorded_chars.update(set(list(x) + list(y_true)))

                    if debug:
                        counter.update("".join(x+y_true))

        logging.info("Saving {} chars to label encoder".format(len(recorded_chars)))
        for char in recorded_chars:
            if char not in self.stoi:
                # Record string for index
                self.stoi[char] = len(self.stoi)
                # Reuse index for string retrieval
                self.itos[self.stoi[char]] = char

        if debug:
            logging.debug(str(counter))

    def readunit(self, line) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """ Read a single line

        :param line:
        :return:
        """
        x = line.strip().split("\t")
        return tuple(x[0]), tuple(x[1])

    def tensorize(self, sentences, padding: Optional[int] = None, batch_first: bool = False) -> torch.Tensor:
        len_sen = (padding or len(sentences[0])) + 2  # 2 = SOS token and EOS token

        # shape [batch_size, len_sentence]
        tensor = torch.tensor(np.array([
            [self.init_token_index] +
            [
                self.stoi.get(char, self.unk_token_index)
                for char in current
            ] + [self.eos_token_index] + [self.pad_token_index] * (len_sen - len(current))
            for current in sentences
        ])).to(self.device)

        if not batch_first:
            # [sentence_len, batch_size]
            return tensor.t()
        return tensor

    def reverse_batch(self, batch, batch_first=False):
        # If dimension is [sentence_len, batch_size]
        if not batch_first:
            batch = batch.t()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [
            [
                self.itos[ind]
                for ind in ex
            ]
            for ex in batch
        ]  # denumericalize
        return batch

    def transcribe_batch(self, batch: List[List[str]]):
        for sentence in batch:
            end = min(len(sentence), sentence.index(self.eos_token))
            yield "".join(sentence[1:end])  # Remove SOS

    def get_dataset(self, path, **kwargs):
        """

        :param path:
        :return:
        """
        return DatasetIterator(self, path, **kwargs)

    @classmethod
    def load(cls, json_content: dict, device) -> "LabelEncoder":
        # pass
        o = cls(device=device, **json_content["params"])
        o.itos = dict({int(i): s for i, s in json_content["itos"].items()})
        o.stoi = json_content["stoi"]
        return o

    def dump(self) -> str:
        return json.dumps({
            "itos": self.itos,
            "stoi": self.stoi,
            "params": {
                "init_token": self.init_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token
            }
        })


if __name__ == "__main__":
    import glob
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    label_encoder = LabelEncoder()
    label_encoder.build(*glob.glob("test_data/*.tsv"), debug=True)

    dataset = label_encoder.get_dataset("test_data/test_encoder.tsv", batch_size=2, random=False)
    print(dataset)

    epoch_batches = dataset.get_epoch()
    x, y = next(epoch_batches)

    assert tuple(x.shape) == (30, 2), "X shape should be right"
    assert tuple(y.shape) == (35, 2), "Y shape should be right"

    assert label_encoder.reverse_batch(y) == [
        ['<SOS>', 's', 'i', ' ', 't', 'e', ' ', 'd', 'e', 's', 'e', 'n', 'í', 'v', 'e', 'r', 'a', 's', ' ', 'p', 'a',
         'r', ' ', 'l', 'e', ' ', 'd', 'o', 'r', 'm', 'i', 'r', '<EOS>', '<PAD>', '<PAD>'],
        ['<SOS>', 'L', 'a', ' ', 'd', 'a', 'm', 'e', ' ', 'h', 'a', 'i', 't', 'é', 'é', ' ', 's', "'", 'e', 'n', ' ',
         'p', 'a', 'r', 't', 'i', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
    ]

    assert list(label_encoder.transcribe_batch(label_encoder.reverse_batch(x))) == [
        "sitedeseníverasparledormir",
        "Ladamehaitéés'enparti"
    ]

    dumped = label_encoder.dump()
    reloaded = LabelEncoder.load(json.loads(dumped), device=DEVICE)
    reloaded.reverse_batch(x)
    assert reloaded.reverse_batch(y) == label_encoder.reverse_batch(y)
