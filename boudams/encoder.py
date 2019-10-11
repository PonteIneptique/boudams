import torch
import torch.cuda
import torch.nn
from typing import Tuple, Dict, List, Optional, Iterator, Sequence, Callable, Union
import logging
import collections
import random
import json
import copy
from operator import itemgetter

from mufidecode import mufidecode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_INIT_TOKEN = "<SOS>"
DEFAULT_EOS_TOKEN = "<EOS>"
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_UNK_TOKEN = "<UNK>"
DEFAULT_MASK_TOKEN = "x"


GT_PAIR = collections.namedtuple("GT", ("x", "x_length", "y", "y_length", "line_index"))


class DatasetIterator:
    def __init__(self, label_encoder: "LabelEncoder", file,
                 batch_size: int = 32,
                 randomized: bool = False):
        self._l_e = label_encoder

        self.encoded: List[GT_PAIR] = []
        self.current_epoch: List[tuple, int] = []
        self.random = randomized
        self.file = file
        self.batch_count = 0
        self.batch_size = batch_size

        self._setup()

    def get_masked(self):
        with open(self.file+".masked", "w") as out_io:
            with open(self.file) as in_io:
                for line in in_io.readlines():
                    if not line.strip():
                        continue

                    x, y = tuple(line.strip().split("\t"))
                    y = "".join([
                        self._l_e.space_token if char == " " else self._l_e.mask_token
                        for char in y
                    ])
                    out_io.write(x+"\t"+y+"\n")
        return DatasetIterator(
            self._l_e, self.file+".masked",
            self.batch_size,
            randomized=self.random
        )

    def __repr__(self):
        return "<DatasetIterator lines='{}' random='{}' \n" \
               "\t batches='{}' batch_size='{}'/>".format(
                    len(self),
                    self.random,
                    self.batch_count,
                    self.batch_size
                )

    def __len__(self):
        """ Number of examples
        """
        return len(self.encoded)

    def _setup(self):
        """ The way this whole iterator works is pretty simple :
        we look at each line of the document, and store its index. This will allow to go directly to this line
        for each batch, without reading the entire file. To do that, we need to read in bytes, otherwise file.seek()
        is gonna cut utf-8 chars in the middle
        """
        logging.info("DatasetIterator reading indexes of lines")
        with open(self.file, "r") as fio:
            for line_index, line in enumerate(fio.readlines()):
                if not line.strip():
                    continue
                x, y = self._l_e.readunit(line.strip())
                self.encoded.append(
                    GT_PAIR(
                        *self._l_e.inp_to_numerical(x),
                        *self._l_e.gt_to_numerical(y),
                        line_index
                    )
                )

        logging.info("DatasetIterator found {} lines in {}".format(len(self), self.file))

        # Get the number of batch for TQDM
        self.batch_count = len(self) // self.batch_size + bool(len(self) % self.batch_size)

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_count = len(self) // self.batch_size + bool(len(self) % self.batch_size)

    def get_epoch(self, device: str = DEVICE, batch_size: int = 32) -> Callable[[], Iterator[Tuple[torch.Tensor, ...]]]:
        # If the batch size is not the original one (most probably is !)
        if batch_size != self.batch_size:
            self.reset_batch_size(batch_size)

        # Create a list of lines
        lines = [] + self.encoded

        # If we need randomization, then DO randomization shuffle of lines
        if self.random is True:
            random.shuffle(lines)

        def iterable():
            for n in range(0, len(lines), self.batch_size):
                xs, y_trues = [], []
                max_len_x, max_len_y = 0, 0  # Needed for padding

                for gt_pair in lines[n:n+self.batch_size]:
                    max_len_x = max(gt_pair.x_length, max_len_x)
                    max_len_y = max(gt_pair.y_length, max_len_y)
                    xs.append(gt_pair.x)
                    y_trues.append(gt_pair.y)

                try:
                    x_tensor, x_length, x_order = self._l_e.pad_and_tensorize(xs, padding=max_len_x, device=device)
                    y_tensor, y_length, y_order = self._l_e.pad_and_tensorize(
                        y_trues,
                        padding=max_len_y,
                        device=device,
                        reorder=x_order
                    )
                except ValueError:
                    print([b.line_index for b in lines[n:n+self.batch_size]])
                    raise
                yield (
                    x_tensor,
                    x_length,
                    y_tensor,
                    y_length
                )

        return iterable


class LabelEncoder:
    def __init__(self,
                 pad_token=DEFAULT_PAD_TOKEN,
                 unk_token=DEFAULT_UNK_TOKEN,
                 mask_token=DEFAULT_MASK_TOKEN,
                 maximum_length: int = None,
                 lower: bool = True,
                 remove_diacriticals: bool = True
                 ):

        self.pad_token: str = pad_token
        self.unk_token: str = unk_token
        self.mask_token: str = mask_token
        self.space_token: str = " "

        self.pad_token_index: int = 2
        self.space_token_index: int = 1
        self.mask_token_index: int = 0
        self.unk_token_index: int = 0  # Put here because it isn't used in masked

        self.max_len: Optional[int] = maximum_length
        self.lower = lower
        self.remove_diacriticals = remove_diacriticals

        self.itos: Dict[int, str] = {
            self.pad_token_index: self.pad_token,
            self.unk_token_index: self.unk_token,
            self.space_token_index: self.space_token
        }  # Id to string for reversal

        self.stoi: Dict[str, int] = {
            self.pad_token: self.pad_token_index,
            self.unk_token: self.unk_token_index,
            self.space_token: self.space_token_index
        }  # String to ID

        # Mask dictionaries
        self.itom: Dict[int, str] = {
            self.pad_token_index: self.pad_token,
            self.mask_token_index: self.mask_token,
            self.space_token_index: self.space_token
        }
        self.mtoi: Dict[str, int] = {
            self.pad_token: self.pad_token_index,
            self.mask_token: self.mask_token_index,
            self.space_token: self.space_token_index
        }

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
            logging.debug(self.stoi)

    def readunit(self, line) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """ Read a single line

        :param line:
        :return:
        """
        inp, out = line.strip().split("\t")
        return tuple(self.prepare(inp)), tuple(self.prepare(out))

    def prepare(self, inp: str):
        if self.remove_diacriticals:
            inp = mufidecode(inp, join=False)
            if self.lower:
                inp = [char.lower() for char in inp]
        else:
            if self.lower:
                inp = inp.lower()

        return inp

    def pad_and_tensorize(
            self,
            sentences: List[List[int]],
            padding: Optional[int] = None,
            reorder: Optional[List[int]] = None,
            device: str = DEVICE
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """ Pad and turn into tensors batches

        :param sentences: List of sentences where characters have been separated into a list and index encoded
        :param padding: padding required (None if every sentence in the same size)
        :param reorder: List of index to reorder the sequence
        :param device: Torch device
        :return: Transformed batch into tensor
        """
        max_len = (padding or len(sentences[0]))

        # From torchtext.data.field ---> lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        # len une fois padding fait dont on soustrait le maximum entre la taille maximum
        # shape [batch_size, len_sentence]
        tensor = []
        lengths = []
        order = []

        original_order = copy.deepcopy(sentences)
        # If GT order was computer, we get it from there
        if reorder:
            sequences = [sentences[index] for index in reorder]
        else:
            sequences = sorted(sentences, key=lambda item: -len(item))

        # Packed sequence need to be in decreasing size order
        for current in sequences:
            order.append(original_order.index(current))
            original_order[order[-1]] = None  # We replace this index with nothing in case some segments are equals
            tensor.append(current + [self.pad_token_index] * (max_len - len(current)))
            lengths.append(len(tensor[-1]) - max(0, max_len - len(current)))

        return torch.tensor(tensor).to(device), torch.tensor(lengths).to(device), order

    def gt_to_numerical(self, sentence: Sequence[str]) -> Tuple[List[int], int]:
        """ Transform GT to numerical

        :param sentence: Sequence of characters (can be a straight string) with spaces
        :return: List of mask indexes
        """
        numericals = [
                self.mask_token_index if ngram[1] != " " else self.space_token_index
                for ngram in zip(*[sentence[i:] for i in range(2)])
                if ngram[0] != " "
            ] + [self.space_token_index]

        return numericals, len(sentence) - sentence.count(" ")

    def inp_to_numerical(self, sentence: Sequence[str]) -> Tuple[List[int], int]:
        """ Transform input sentence to numerical

        :param sentence: Sequence of characters (can be a straight string) without spaces
        :return: List of character indexes
        """
        return (
            [self.stoi.get(char, self.unk_token_index) for char in sentence],
            len(sentence)
        )

    def reverse_batch(
            self,
            batch: Union[list, torch.Tensor],
            ignore: Optional[Tuple[str, ...]] = None,
            masked: Optional[Union[list, torch.Tensor]] = None
    ):
        ignore = ignore or ()
        # If dimension is [sentence_len, batch_size]
        if not isinstance(batch, list):

            with torch.cuda.device_of(batch):
                batch = batch.tolist()

        if masked is not None:
            if not isinstance(masked, list):
                with torch.cuda.device_of(masked):
                    masked = masked.tolist()

            if not isinstance(masked[0][0], str):
                masked = [
                    [
                        self.itos[masked_token]
                        for masked_token in sentence
                    ]
                    for sentence in masked
                ]
            else:
                masked = [
                    list(sentence)
                    for sentence in masked
                ]
            print(ignore)

            return [
                [
                    tok
                    for masked_token, mask_token in zip(masked_sentence, space_mask)
                    if space_mask not in ignore and masked_token not in ignore
                    for tok in [masked_token] + ([" "] if mask_token == self.space_token_index else [])
                ]
                for masked_sentence, space_mask in zip(masked, batch)
            ]

        if ignore is True:
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
            yield "".join(sentence).strip()  # Remove SOS

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
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
                "remove_diacriticals": self.remove_diacriticals,
                "lower": self.lower
            }
        })


if __name__ == "__main__":
    import glob
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    label_encoder = LabelEncoder()
    label_encoder.build(*glob.glob("test_data/*.tsv"), debug=True)

    dataset = label_encoder.get_dataset("test_data/test_encoder.tsv", randomized=False)

    epoch_batches = dataset.get_epoch(batch_size=2)()
    x, x_len, y, y_len = next(epoch_batches)

    assert tuple(x.shape) == (2, 28), "X shape should be (28, 2), got {}".format(tuple(x.shape))
    assert tuple(y.shape) == (2, 33), "Y shape should be right"

    assert label_encoder.reverse_batch(y) == [
        [
            '<SOS>', 's', 'i', ' ', 't', 'e', ' ', 'd', 'e', 's', 'e', 'n', 'i', 'v', 'e', 'r', 'a', 's', ' ', 'p', 'a',
            'r', ' ', 'l', 'e', ' ', 'd', 'o', 'r', 'm', 'i', 'r', '<EOS>']
        ,
        [
            '<SOS>', 'l', 'a', ' ', 'd', 'a', 'm', 'e', ' ', 'h', 'a', 'i', 't', 'e', 'e', ' ', 's', "'", 'e', 'n', ' ',
            'p', 'a', 'r', 't', 'i', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'
        ]
    ]

    assert list(label_encoder.transcribe_batch(label_encoder.reverse_batch(x))) == [
        "sitedeseniverasparledormir",
        "ladamehaitees'enparti"
    ]

    dumped = label_encoder.dump()
    reloaded = LabelEncoder.load(json.loads(dumped))
    reloaded.reverse_batch(x)
    assert reloaded.reverse_batch(y) == label_encoder.reverse_batch(y)

    assert len(list(epoch_batches)) == 2, "There should be only to more batches"

    # Masked
    old_y = y
    label_encoder = LabelEncoder(masked=True)
    label_encoder.build(*glob.glob("test_data/*.tsv"), debug=True)

    dataset = label_encoder.get_dataset("test_data/test_encoder.tsv", randomized=False)

    epoch_batches = dataset.get_epoch(batch_size=5)()
    x, _, y, _ = next(epoch_batches)

    assert x.shape == y.shape, "OTHERWISE WE'RE SCREWED"

    # Somehow, although stuff IS padded and each sequence should have the same size, this is not the case...
    # I definitely need to spleep on it
    reversed_data = list(label_encoder.reverse_batch(y, masked=x))

    assert [
               l
               for l in list(map(lambda liste: len(liste)-liste.count(" "), reversed_data))
               if l != len(reversed_data[0]) - reversed_data[0].count(" ")
           ] == [], \
        "All element should have the same size (length : %s) if we remove the spaces" % list(map(len, reversed_data))
    print(reversed_data)
    assert reversed_data == [
        ['<SOS>', 'e', ' ', 'd', 'e', 'u', 's', ' ', 't', 'u', 'n', 'e', 'i', 'r', 'e', ' ', 'e', ' ', 'p', 'l', 'u',
         'i', 'e', ' ', 'm', 'e', 'r', 'v', 'e', 'i', 'l', 'l', 'u', 's', 'e', ' ', 'a', ' ', 'c', 'e', 'l', ' ', 'j',
         'u', 'r', ' ', 'e', 'n', 'v', 'e', 'i', 'a', 'd', '<EOS>'],
        ['<SOS>', 's', 'i', ' ', 't', 'e', ' ', 'd', 'e', 's', 'e', 'n', 'i', 'v', 'e', 'r', 'a', 's', ' ', 'p', 'a',
         'r', ' ', 'l', 'e', ' ', 'd', 'o', 'r', 'm', 'i', 'r', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
        ['<SOS>', 'l', 'a', ' ', 'd', 'a', 'm', 'e', ' ', 'h', 'a', 'i', 't', 'e', 'e', ' ', 's', "'", 'e', 'n', ' ',
         'p', 'a', 'r', 't', 'i', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>'],
        ['<SOS>', 'e', ' ', 'm', 'a', 'n', 'j', 'a', 'd', ' ', 'l', 'a', ' ', 'c', 'h', 'a', 'r',
         ' ', 'o', 'd', ' ', 'l', 'e', ' ', 's', 'a', 'n', 'c', '<EOS>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
        ['<SOS>', 'e', ' ', 'a', ' ', 's', 'u', 'n', ' ', 's', 'e', 'r', 'v', 'i', 's', 'e', ' ', 'l', 'e', 's', ' ',
         'm', 'e', 't', 'r', 'a', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
         '<PAD>', '<PAD>', '<PAD>']
    ]

