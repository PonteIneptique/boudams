import random
import re
from typing import Dict, Optional, Tuple, Sequence, List

import torch
from boudams.utils import parse_params

DEFAULT_PAD_TOKEN = "垫"
DEFAULT_MASK_TOKEN = "-"
DEFAULT_WB_TOKEN = "|"
DEFAULT_REMOVE_TOKEN = "⌫"
DEFAULT_ORIGINAL_TOKEN = ""


class MaskValueException(Exception):
    """ Exception raised when a token is longer than a character """


class MaskGenerationError(Exception):
    """ Exception raised when a mask is not of the same size as the input transformed string """


class SimpleSpaceMode:
    NormalizeSpace: bool = True

    def __init__(self, masks: Dict[str, int] = None):
        self.name = "simple-space"
        self.masks_to_index: Dict[str, int] = masks or {
            DEFAULT_PAD_TOKEN: 0,
            DEFAULT_MASK_TOKEN: 1,
            DEFAULT_WB_TOKEN: 2
        }
        self.index_to_mask: Dict[str, int] = {
            value: key
            for value, key in self.masks_to_index.items()
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
                    raise MaskValueException(
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
        ("j'aiuncheval", '---|-|-----|')
        """
        split = self._space.split(string.strip())
        masks = DEFAULT_WB_TOKEN.join([DEFAULT_MASK_TOKEN * (len(tok)-1) for tok in split]) + DEFAULT_WB_TOKEN
        model_input = "".join(split)
        assert len(masks) == len(model_input), f"Length of input and mask should be equal `{masks}` + `{model_input}` + `{string}`"
        return model_input, masks

    def encode_mask(self, masked_string: Sequence[str]) -> List[int]:
        """ Encodes into a list of index a string

        :param masked_string: String masked by the current Mode
        :return: Pre-tensor input

        >>> (SimpleSpaceMode()).encode_mask("xxx|x|xxxxx|")
        [1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2]
        """
        return [self.masks_to_index[char] for char in masked_string]

    def apply_mask_to_string(self, input_string: List[str], masks: List[int]) -> str:
        def apply():
            for char, mask in zip(input_string, masks):
                if mask == self.pad_token_index:
                    break
                if self.index_to_masks_name[mask] == "WB":
                    yield char + " "
                else:
                    yield char
        return "".join(apply())

    def prepare_input(self, string: str) -> str:
        return self._space.sub("", string)

    def _clean_matrix(self, confusion_matrix, pad_token_index):
        indexes = torch.tensor([
            i
            for i in range(self.classes_count)
            if i != pad_token_index
        ]).type_as(confusion_matrix)

        return confusion_matrix[indexes][:, indexes]

    def computer_wer(self, confusion_matrix):
        clean_matrix = self._clean_matrix(confusion_matrix, self.pad_token_index)

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


class AdvancedSpaceMode(SimpleSpaceMode):
    def __init__(self, masks: Dict[str, int] = None):
        self.name = "advanced-space"
        self.masks_to_index: Dict[str, int] = masks or {
            DEFAULT_PAD_TOKEN: 0,
            DEFAULT_MASK_TOKEN: 1,
            DEFAULT_WB_TOKEN: 2,
            DEFAULT_REMOVE_TOKEN: 3,
            DEFAULT_ORIGINAL_TOKEN: 4
        }
        self.index_to_mask: Dict[str, int] = masks or {
            0: DEFAULT_PAD_TOKEN,
            1: DEFAULT_MASK_TOKEN,
            2: DEFAULT_WB_TOKEN,
            3: DEFAULT_REMOVE_TOKEN,
            4: DEFAULT_ORIGINAL_TOKEN
        }
        self.index_to_masks_name: Dict[int, str] = {
            0: "PAD",
            1: "W",
            2: "WB",
            3: "REMOVE",
            4: "ORIGINAL"
        }
        self.masks_name_to_index: Dict[str, int] = {
            "PAD": 0,
            "W": 1,
            "WB": 2,
            "REMOVE": 3,
            "ORIGINAL": 4
        }
        self.pad_token = DEFAULT_PAD_TOKEN
        self._pad_token_index = self.masks_to_index[self.pad_token]
        self._space = re.compile(r"\s+")

        self._check()

    def _check(self):
        for char in self.masks_to_index:
            if char != self.pad_token:  # We do not limit <PAD> to a single char because it's not dumped in a string
                if len(char) != 1:
                    raise MaskValueException(
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

        >>> (AdvancedSpaceMode()).generate_mask("j'ai un cheval", tokens_ratio={"fake-space": 1, 'keep-space': 0})
        ("j ' a iu nc h e v a l", '-⌫-⌫-⌫|-⌫|-⌫-⌫-⌫-⌫-⌫|')

        >>> (AdvancedSpaceMode()).generate_mask("j'ai un cheval", tokens_ratio={"fake-space": 0, 'keep-space': 1})
        ("j'ai un cheval", '---|-|-----|')
        """

        model_input: List[str] = []
        masks: List[str] = []
        string = string.strip()
        for char, next_char in zip(string, string[1:]+" "):
            if char.strip():  # It's not a space
                model_input.append(char)
                masks.append(DEFAULT_MASK_TOKEN)
                if next_char.strip() and random.random() < tokens_ratio.get("fake-space", 0):
                    model_input.append(" ")
                    masks.append(DEFAULT_REMOVE_TOKEN)
            else:
                if len(masks):
                    masks[-1] = DEFAULT_WB_TOKEN
                    if random.random() < tokens_ratio.get("keep-space", 0):
                        model_input.append(" ")  # Space are normalized
                        masks.append(DEFAULT_ORIGINAL_TOKEN)
        masks[-1] = DEFAULT_WB_TOKEN
        assert len(masks) == len(model_input), f"Length of input and mask should be equal `{masks}` + `{model_input}`"
        return "".join(model_input), "".join(masks)

    def encode_mask(self, masked_string: Sequence[str]) -> List[int]:
        """ Encodes into a list of index a string

        :param masked_string: String masked by the current Mode
        :return: Pre-tensor input

        >>> (AdvancedSpaceMode()).encode_mask("-⌫--|-|-|")
        [1, 3, 1, 1, 2, 1, 2, 4, 1, 2]
        """
        return [self.masks_to_index[char] for char in masked_string]

    def apply_mask_to_string(self, input_string: str, masks: List[int]) -> str:
        """ Apply a prediction to a string

        :param input_string:
        :param masks:
        :return:

        >>> (AdvancedSpaceMode()).apply_mask_to_string("J 'aiun nu", [1, 3, 1, 1, 2, 1, 2, 4, 1, 2])
        "J'ai un nu"
        """
        def apply():
            for char, mask in zip(input_string, masks):
                if mask == self.pad_token_index:
                    break
                if self.index_to_masks_name[mask] == "WB":
                    yield char + " "
                elif self.index_to_masks_name[mask] == "REMOVE":
                    continue
                else:
                    yield char
        return self._space.sub(" ", "".join(apply())).strip()

    def prepare_input(self, string: str) -> str:
        return self._space.sub(" ", string).strip()

    def computer_wer(self, confusion_matrix):
        clean_matrix = self._clean_matrix(confusion_matrix, self.pad_token_index)

        space_tokens = [
            space_index if space_index < self.pad_token_index else space_index-1
            for space_index in [
                self.masks_to_index[DEFAULT_WB_TOKEN],
                self.masks_to_index[DEFAULT_REMOVE_TOKEN],
                self.masks_to_index[DEFAULT_REMOVE_TOKEN]
            ]
        ]

        nb_space_gt = (
            clean_matrix[space_tokens].sum() +
            clean_matrix[:, space_tokens].sum() -
            clean_matrix[space_tokens, space_tokens].sum()
        )

        nb_missed_space = clean_matrix.sum() - torch.diagonal(clean_matrix, 0).sum()
        return nb_missed_space / nb_space_gt
