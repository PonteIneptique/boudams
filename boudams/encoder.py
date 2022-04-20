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

from boudams.modes import SimpleSpaceMode, AdvancedSpaceMode

DEFAULT_INIT_TOKEN = "<SOS>"
DEFAULT_EOS_TOKEN = "<EOS>"
DEFAULT_UNK_TOKEN = "<UNK>"


class LabelEncoder:
    Modes = {
        "simple-space": SimpleSpaceMode,
        "advanced-space": AdvancedSpaceMode
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
            0: self.pad_token,
            self.unk_token_index: unk_token
        }  # String to ID

        self.stoi: Dict[str, int] = {
            self.pad_token: self.pad_token_index,
            self.unk_token: self.unk_token_index
        }  # String to ID

        # Mask dictionaries
        self.itom: Dict[int, str] = dict([
            (tok_id, mask) for (mask, tok_id) in self._mode.masks_to_index.items()
        ])
        self.mtoi: Dict[str, int] = dict([
            (mask, tok_id) for (mask, tok_id) in self._mode.masks_to_index.items()
        ])

    @property
    def mask_count(self):
        return len(self.mtoi)

    @property
    def mode(self):
        return self._mode

    def __len__(self):
        return len(self.stoi)

    def build(self, train, *paths, debug=False) -> int:
        """ Builds vocabulary

        :param paths: Path of file to read
        :return: Maximum sentence size
        """
        recorded_chars = set()
        counter = None
        if debug:
            counter = collections.Counter()

        logging.info("Reading files for vocabulary building")
        max_sentence_size = 0
        for path_idx, path in enumerate([train, *paths]):
            with open(path) as fio:
                for line in fio.readlines():
                    x, _ = self.readunit(line)
                    seq_len = len(x)
                    if seq_len > max_sentence_size:
                        max_sentence_size = seq_len
                    recorded_chars.update(set(list(x)))

        logging.info("Saving {} chars to label encoder".format(len(recorded_chars)))
        for char in recorded_chars:
            if char not in self.stoi:
                # Record string for index
                self.stoi[char] = len(self.stoi)
                # Reuse index for string retrieval
                self.itos[self.stoi[char]] = char

        return max_sentence_size

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

    def numerical_to_sent(self, encoded_sentence: List[int]) -> List[str]:
        """ Transform a list of integers to a string

        :param encoded_sentence: List of index
        :return: Characters
        """
        return [
            self.itos[char_idx]
            for char_idx in encoded_sentence
            if char_idx != self.pad_token_index
        ]

    def reverse_batch(
        self,
        input_batch: Union[List[List[int]], torch.Tensor],
        mask_batch: Optional[Union[list, torch.Tensor]],
        override_numerical_input: Optional[List[str]] = None
    ):
        """ Produce result strings for a batch

        :param input_batch: Input batch
        :param mask_batch: Output batch
        :return: List of strings with applied masks
        """
        # If we override the source
        if not override_numerical_input:
            if not isinstance(input_batch, list):
                with torch.cuda.device_of(input_batch):
                    input_batch = input_batch.tolist()
        if not isinstance(mask_batch, list):
            with torch.cuda.device_of(mask_batch):
                mask_batch = mask_batch.tolist()

        return [
            self.mode.apply_mask_to_string(
                input_string=self.numerical_to_sent(batch_seq) if not isinstance(batch_seq, str) else batch_seq,
                masks=masked_seq
            )
            for batch_seq, masked_seq in zip(override_numerical_input or input_batch, mask_batch)
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
        o = cls(mode=json_content["mode"], **json_content["params"])

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

        col_pad = header.index(self.mode.index_to_masks_name.get(self.pad_token_index))
        return tabulate.tabulate(
            [
                row[:col_pad] + row[col_pad+1:]
                for (col_id, row) in enumerate(confusion)
                if row[0] != "PAD"
            ],
            headers=[col for col in header if col != "PAD"]
        )
