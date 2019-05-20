import torch
import torch.cuda
import torch.nn
from typing import Tuple, Dict, List, Optional, Iterator, Sequence, Callable, Union
import logging
import collections
import random
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

    def __len__(self):
        return len(self.line_starts_offsets)

    def _setup(self):
        """ The way this whole iterator works is pretty simple :
        we look at each line of the document, and store its index. This will allow to go directly to this line
        for each batch, without reading the entire file. To do that, we need to read in bytes, otherwise file.seek()
        is gonna cut utf-8 chars in the middle
        """
        logging.info("DatasetIterator reading indexes of lines")
        with open(self.file, "rb") as fio:
            offset = 0
            for line in fio:
                if line.strip():  # if line is not empty
                    self.line_starts_offsets.append(offset)
                offset += len(line)

        logging.info("DatasetIterator found {} lines in {}".format(self.length, self.file))

        # Get the number of batch for TQDM
        self.batch_count = self.length // self.batch_size + bool(self.length % self.batch_size)

    @property
    def length(self):
        return len(self.line_starts_offsets)

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_count = self.length // self.batch_size + bool(self.length % self.batch_size)

    def get_line(self, *line_index):
        with open(self.file, "rb") as fio:
            for line_start in line_index:
                fio.seek(line_start)
                yield self._l_e.readunit(fio.readline().decode("utf-8").strip())

    def get_epoch(self, device: str = DEVICE, batch_size: int = 32) -> Callable[[], Iterator[Tuple[torch.Tensor, ...]]]:
        # If the batch size is not the original one (most probably is !)
        if batch_size != self.batch_size:
            self.reset_batch_size(batch_size)

        # Create a list of lines
        lines = [] + self.line_starts_offsets

        # If we need randomization, then DO randomization shuffle of lines
        if self.random:
            random.shuffle(lines)

        def iterable():
            for n in range(0, len(lines), self.batch_size):
                xs, y_trues = [], []
                max_len_x, max_len_y = 0, 0  # Needed for padding

                for x, y in self.get_line(*lines[n:n+self.batch_size]):
                    max_len_x = max(len(x), max_len_x)
                    max_len_y = max(len(y), max_len_y)
                    xs.append(x)
                    y_trues.append(y)

                yield (
                    *self._l_e.tensorize(xs, padding=max_len_x, device=device),
                    *self._l_e.tensorize(y_trues, padding=max_len_y, device=device)
                )

        return iterable



class LabelEncoder:
    def __init__(self,
                 init_token=DEFAULT_INIT_TOKEN,
                 eos_token=DEFAULT_EOS_TOKEN,
                 pad_token=DEFAULT_PAD_TOKEN,
                 unk_token=DEFAULT_UNK_TOKEN,
                 maximum_length: int = None
                 ):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.init_token_index = 0
        self.eos_token_index = 1
        self.pad_token_index = 2
        self.unk_token_index = 3
        self.max_len: Optional[int] = maximum_length
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

    def __len__(self):
        return len(self.stoi)

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
        if self.max_len and len(x[0]) >= len(x[1]) > self.max_len:
            raise AssertionError("Data should be smaller than maximum length")
        return tuple(x[0]), tuple(x[1])

    def tensorize(self,
                  sentences: List[Sequence[str]],
                  padding: Optional[int] = None,
                  batch_first: bool = False,
                  device: str = DEVICE) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Transform a list of sentences into a batched Tensor with its tensor size

        :param sentences: List of sentences where characters have been separated into a list
        :param padding: padding required (None if every sentence in the same size)
        :param batch_first: Whether we need [batch_size, sentence_len] instead of [sentence_len, batch_size]
        :param device: Torch device
        :return: Transformed batch into tensor
        """
        max_len = (padding or len(sentences[0]))

        # From torchtext.data.field ---> lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        # len une fois padding fait dont on soustrait le maximum entre la taille maximum
        # shape [batch_size, len_sentence]
        tensor = []
        lengths = []

        obligatory_tokens = 2  # Tokens for init and end of string

        # Packed sequence need to be in decreasing size order
        for current in sorted(sentences, key=lambda item: -len(item)):
            tensor.append(
                # A sentence start with an init token
                [self.init_token_index] +
                # It's followed by each char index in the vocabulary
                [self.stoi.get(char, self.unk_token_index) for char in current] +
                # Then we get the end of sentence milestone
                [self.eos_token_index] +
                # And we padd for what remains
                [self.pad_token_index] * (max_len + obligatory_tokens - len(current))  # 2 = SOS token and EOS token
            )
            lengths.append(len(tensor[-1]) - max(0, max_len - len(current)))

        tensor = torch.tensor(tensor).to(device)

        if not batch_first:
            # [sentence_len, batch_size]
            return tensor.t(), torch.tensor(lengths).to(device)
        return tensor, torch.tensor(lengths).to(device)

    def reverse_batch(self, batch: Union[list, torch.Tensor], batch_first=False,
                      ignore: Optional[Tuple[str, ...]] = None):
        # If dimension is [sentence_len, batch_size]
        if not isinstance(batch, list):
            if not batch_first:
                batch = batch.t()

            with torch.cuda.device_of(batch):
                batch = batch.tolist()

        if ignore:
            batch = [
                [
                    self.itos[ind]
                    for ind in ex
                    if ind not in ignore
                ]
                for ex in batch
            ]
        else:
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
            end = len(sentence) if self.eos_token not in sentence else sentence.index(self.eos_token)
            yield "".join(sentence[1:end])  # Remove SOS

    def get_dataset(self, path, **kwargs):
        """

        :param path:
        :return:
        """
        return DatasetIterator(self, path, **kwargs)

    @classmethod
    def load(cls, json_content: dict) -> "LabelEncoder":
        # pass
        logging.info("Loading LabelEncoder")
        o = cls(**json_content["params"])
        o.itos = dict({int(i): s for i, s in json_content["itos"].items()})
        o.stoi = json_content["stoi"]
        logging.info("Loaded LabelEncoder with {} tokens".format(len(o.itos) - 4))
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

    dataset = label_encoder.get_dataset("test_data/test_encoder.tsv", random=False)

    epoch_batches = dataset.get_epoch(batch_size=2)
    x, x_len, y, y_len = next(epoch_batches)

    assert tuple(x.shape) == (30, 2), "X shape should be (30, 2), got {}".format(tuple(x.shape))
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
    reloaded = LabelEncoder.load(json.loads(dumped))
    reloaded.reverse_batch(x)
    assert reloaded.reverse_batch(y) == label_encoder.reverse_batch(y)

    assert len(list(epoch_batches)) == 2, "There should be only to more batches"
