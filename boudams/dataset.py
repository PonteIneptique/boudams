import logging
import collections
from typing import List, Tuple
from operator import itemgetter

import torch.utils.data as torch_data

from boudams.encoder import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence

GT_PAIR = collections.namedtuple("GT", ("x", "x_length", "y", "y_length", "line_index"))


class BoudamsDataset(torch_data.Dataset):
    def __init__(
        self,
        label_encoder: "LabelEncoder",
        *files: str
    ):
        self._l_e = label_encoder
        self.encoded: List[GT_PAIR] = []
        self.files: Tuple[str] = files
        self._setup()

    def __repr__(self):
        return f"<BoudamsDataset lines='{len(self)}'/>"

    def __len__(self):
        """ Number of examples
        """
        return len(self.encoded)

    def __getitem__(self, item):
        return self.encoded[item]

    def _setup(self):
        """ The way this whole iterator works is pretty simple :
        we look at each line of the document, and store its index. This will allow to go directly to this line
        for each batch, without reading the entire file. To do that, we need to read in bytes, otherwise file.seek()
        is gonna cut utf-8 chars in the middle
        """
        logging.info("DatasetIterator reading indexes of lines")
        for file in self.files:
            with open(file, "r") as fio:
                for line_index, line in enumerate(fio.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    x, y = self._l_e.readunit(line)
                    self.encoded.append(
                        GT_PAIR(
                            *self._l_e.inp_to_numerical(x),
                            *self._l_e.gt_to_numerical(y),
                            f"File:{file}#Line:{line_index}"
                        )
                    )

        logging.info("DatasetIterator found {} lines in {}".format(len(self), ", ".join(self.files)))

    def train_collate_fn(self, batch: List[GT_PAIR]):
        """
        DataLoaderBatch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        batch_size = len(batch)
        x, x_length, y, y_length, _ = list(zip(*sorted(batch, key=itemgetter(1), reverse=True)))
        return (
            pad_sequence([torch.tensor(x_i) for x_i in x]),
            torch.tensor(x_length),
            pad_sequence([torch.tensor(y_i) for y_i in y])#,
            #torch.tensor(x_length)
        )