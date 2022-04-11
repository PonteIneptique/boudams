import re

import tabulate
import torch
import torch.cuda
import torch.nn
from typing import Tuple, Dict, List, Optional, Sequence, Union
import logging
import collections
import json
import copy
import tabulate

from mufidecode import mufidecode

DEFAULT_INIT_TOKEN = "<SOS>"
DEFAULT_EOS_TOKEN = "<EOS>"
DEFAULT_PAD_TOKEN = "垫"
DEFAULT_UNK_TOKEN = "<UNK>"
DEFAULT_MASK_TOKEN = "x"
DEFAULT_WB_TOKEN = "|"


class SimpleSpaceMode:
    NormalizeSpace: bool = True

    class MaskValueException(Exception):
        """ Exception raised when a token is longer than a character """

    class MaskGenerationError(Exception):
        """ Exception raised when a mask is not of the same size as the input transformed string """

    def __init__(self, masks: Dict[str, int] = None):
        self.name = "Default"
        self.masks_to_index: Dict[str, int] = masks or {
            DEFAULT_PAD_TOKEN: 0,
            DEFAULT_MASK_TOKEN: 1,
            DEFAULT_WB_TOKEN: 2
        }
        self.index_to_mask: Dict[str, int] = masks or {
            0: DEFAULT_PAD_TOKEN,
            1: DEFAULT_MASK_TOKEN,
            2: DEFAULT_WB_TOKEN
        }
        self.index_to_masks_name: Dict[int, str] = {
            0: "PAD",
            1: "W",
            2: "WB"
        }
        self.masks_name_to_index: Dict[str, int] = {
            "PAD": 0,
            "W": 1,
            "WB": 2
        }
        self.pad_token = DEFAULT_PAD_TOKEN
        self._pad_token_index = self.masks_to_index[self.pad_token]
        self._space = re.compile(r"\s")

        self._check()

    def _check(self):
        for char in self.masks_to_index:
            if char != self.pad_token:  # We do not limit <PAD> to a single char because it's not dumped in a string
                if len(char) != 1:
                    raise SimpleSpaceMode.MaskValueException(
                        f"Mask characters cannot be longer than one char "
                        f"(Found: `{char}` "
                        f"for {self.index_to_masks_name[self.masks_to_index[char]]})")

    @property
    def pad_token_index(self) -> int:
        return self._pad_token_index

    @property
    def classes_count(self):
        return len(self.masks_to_index)

    def generate_mask(
            self,
            string: str,
            tokens_ratio: Optional[Dict[str, float]] = None
    ) -> Tuple[str, str]:
        """ From a well-formed ground truth input, generates a fake error-containing string

        :param string: Input string
        :param tokens_ratio: Dict of TokenName
        :return:

        >>> (SimpleSpaceMode()).generate_mask("j'ai un cheval")
        ('xxx|x|xxxxx|', "j'aiuncheval")
        """
        split = self._space.split(string)
        masks = DEFAULT_WB_TOKEN.join([DEFAULT_MASK_TOKEN * (len(tok)-1) for tok in split]) + DEFAULT_WB_TOKEN
        model_input = "".join(split)
        assert len(masks) == len(model_input), f"Length of input and mask should be equal `{masks}` + `{model_input}`"
        return model_input, masks

    def encode_mask(self, masked_string: Sequence[str]) -> List[int]:
        """ Encodes into a list of index a string

        :param masked_string: String masked by the current Mode
        :return: Pre-tensor input

        >>> (SimpleSpaceMode()).encode_mask("xxx|x|xxxxx|")
        [1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2]
        """
        return [self.masks_to_index[char] for char in masked_string]

    def apply_mask_to_string(self, input_string: str, masked_string: List[int]) -> str:
        def apply():
            for char, mask in zip(input_string, masked_string):
                if mask == self.pad_token_index:
                    break
                if self.index_to_masks_name[mask] == "WB":
                    yield char + " "
                else:
                    yield char
        return "".join(apply())

    def prepare_input(self, string: str) -> str:
        return self._space.sub("", string)

    def computer_wer(self, confusion_matrix):
        indexes = torch.tensor([
            i
            for i in range(self.classes_count)
            if i != self.pad_token_index
        ]).type_as(confusion_matrix)

        clean_matrix = confusion_matrix[indexes][:, indexes]
        space_token_index = self.masks_to_index[DEFAULT_WB_TOKEN]
        if space_token_index > self.pad_token_index:
            space_token_index -= 1
        nb_space_gt = (
            clean_matrix[space_token_index].sum() +
            clean_matrix[:, space_token_index].sum() -
            clean_matrix[space_token_index, space_token_index]
        )

        nb_missed_space = clean_matrix.sum() - torch.diagonal(clean_matrix, 0).sum()
        return nb_missed_space / nb_space_gt


class LabelEncoder:
    Modes = {
        "simple-space": SimpleSpaceMode
    }

    # For test purposes
    EXAMPLE_LINE = "\t".join(['a b c D', 'x x x x'])

    def __init__(
        self,
        mode: str = "simple-space",
        maximum_length: int = None,
        lower: bool = True,
        remove_diacriticals: bool = True,
        unk_token: str = DEFAULT_UNK_TOKEN
    ):
        self._mode_string: str = mode
        self._mode: SimpleSpaceMode = self.Modes[mode]()

        self.pad_token: str = self._mode.pad_token
        self.pad_token_index: int = 0  # Only for CHARS

        self.unk_token: str = unk_token
        self.unk_token_index: int = 1

        self.max_len: Optional[int] = maximum_length
        self.lower = lower
        self.remove_diacriticals = remove_diacriticals

        self.itos: Dict[int, str] = {
            self.pad_token: 0,
            self.unk_token: self.unk_token_index
        }  # String to ID

        self.stoi: Dict[str, int] = {
            self.pad_token: self.pad_token_index,
            self.unk_token: self.unk_token_index
        }  # String to ID

        # Mask dictionaries
        self.itom: Dict[int, str] = dict([
            (tok_id, mask) for (mask, tok_id) in self._mode.masks_to_index.items()
        ])
        self.mtoi: Dict[int, str] = dict([
            (tok_id, mask) for (mask, tok_id) in self._mode.masks_to_index.items()
        ])

    @property
    def mask_count(self):
        return len(self.mtoi)

    @property
    def mode(self):
        return self._mode

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
                    x, _ = self.readunit(line)
                    recorded_chars.update(set(list(x)))

        logging.info("Saving {} chars to label encoder".format(len(recorded_chars)))
        for char in recorded_chars:
            if char not in self.stoi:
                # Record string for index
                self.stoi[char] = len(self.stoi)
                # Reuse index for string retrieval
                self.itos[self.stoi[char]] = char

    def readunit(self, line) -> Tuple[str, str]:
        """ Read a single line

        :param line:
        :return:

        >>> (LabelEncoder(lower=True)).readunit(LabelEncoder.EXAMPLE_LINE)
        (('a', ' ', 'b', ' ', 'c', ' ', 'd'), 'x x x x')
        """
        inp, out = line.strip().split("\t")
        return self.prepare(inp), out

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
            reorder: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """ Pad and turn into tensors batches

        :param sentences: List of sentences where characters have been separated into a list and index encoded
        :param padding: padding required (None if every sentence in the same size)
        :param reorder: List of index to reorder the sequence
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

        return torch.tensor(tensor), torch.tensor(lengths), order

    def mask_to_numerical(self, sentence: Sequence[str]) -> Tuple[List[int], int]:
        """ Transform GT to numerical

        :param sentence: Sequence of characters (can be a straight string) with spaces
        :return: List of mask indexes
        """
        return self.mode.encode_mask(sentence), len(sentence)

    def sent_to_numerical(self, sentence: Sequence[str]) -> Tuple[List[int], int]:
        """ Transform input sentence to numerical

        :param sentence: Sequence of characters (can be a straight string) without spaces
        :return: List of character indexes
        """
        return (
            [self.stoi.get(char, self.unk_token_index) for char in sentence],
            len(sentence)
        )

    def numerical_to_sent(self, encoded_sentence: List[int]) -> str:
        """ Transform a list of integers to a string

        :param encoded_sentence: List of index
        :return: Characters
        """
        return "".join([
            self.itos[char_idx]
            for char_idx in encoded_sentence
            if char_idx != self.pad_token
        ])

    def reverse_batch(
        self,
        batch: Union[list, torch.Tensor],
        mask_batch: Optional[Union[list, torch.Tensor]] = None
    ):
        """ Produce result strings for a batch

        :param batch: Input batch
        :param mask_batch: Output batch
        :return: List of strings with applied masks
        """
        # If dimension is [sentence_len, batch_size]
        if not isinstance(batch, list):
            with torch.cuda.device_of(batch):
                batch = batch.tolist()
        if not isinstance(mask_batch, list):
            with torch.cuda.device_of(mask_batch):
                mask_batch = mask_batch.tolist()

        return [
            self.mode.apply_mask_to_string(
                input_string=self.numerical_to_sent(batch_seq),
                masked_string=masked_seq
            )
            for batch_seq, masked_seq in zip(batch, mask_batch)
        ]

    def transcribe_batch(self, batch: List[List[str]]):
        for sentence in batch:
            yield "".join(sentence).strip()  # Remove SOS

    def get_dataset(self, *path):
        """

        :param path:
        :return:
        """
        from boudams.dataset import BoudamsDataset

        return BoudamsDataset(self, *path)

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
            "mode": self._mode_string,
            "params": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "remove_diacriticals": self.remove_diacriticals,
                "lower": self.lower
            }
        })

    def format_confusion_matrix(self, confusion: List[List[int]]):
        header = [
            "",
            *[
                self.mode.index_to_masks_name.get(index, index)
                for index in sorted(list(self.itom.keys()))
            ]
        ]
        confusion = [
            [head, *list(map(str, confusion_row))]
            for head, confusion_row in zip(header[1:], confusion)
        ]

        col_pad = header.index(self.pad_token)
        return tabulate.tabulate(
            [
                row[:col_pad] + row[col_pad+1:]
                for (col_id, row) in enumerate(confusion)
                if row[0] != self.pad_token
            ],
            headers=[col for col in header if col != self.pad_token]
        )


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
    reversed_data = list(label_encoder.reverse_batch(y, mask_batch=x))

    assert [
               l
               for l in list(map(lambda liste: len(liste)-liste.count(" "), reversed_data))
               if l != len(reversed_data[0]) - reversed_data[0].count(" ")
           ] == [], \
        "All element should have the same size (length : %s) if we remove the spaces" % list(map(len, reversed_data))
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

