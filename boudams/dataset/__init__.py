import random
from typing import List, Iterator, Tuple, Union
from torchtext.data import Field, ReversibleField, TabularDataset, Dataset, Iterator as TorchIterator, Example

DEFAULT_INIT_TOKEN = "<SOS>"
DEFAULT_EOS_TOKEN = "<EOS>"
DEFAULT_PAD_TOKEN = "<PAD>"

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%20and%20Inference.ipynb
#  Include_lengths is used only on source normally, need to be safe on this
CharacterField: ReversibleField = ReversibleField(tokenize=list, init_token=DEFAULT_INIT_TOKEN,
                                                  eos_token=DEFAULT_EOS_TOKEN, pad_token=DEFAULT_PAD_TOKEN,
                                                  lower=False, include_lengths=True)


def get_datasets(train, test, dev) -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
    fields = [("src", CharacterField), ("trg", CharacterField)]
    return (
        TabularDataset(train, format="TSV", fields=fields, skip_header=False, csv_reader_params={"quotechar": '@'}),
        TabularDataset(test, format="TSV", fields=fields, skip_header=False, csv_reader_params={"quotechar": '@'}),
        TabularDataset(dev, format="TSV", fields=fields, skip_header=False, csv_reader_params={"quotechar": '@'})
    )


def build_vocab(field: ReversibleField, datasets: Iterator[TabularDataset]) -> ReversibleField:
    for dataset in datasets:
        field.build_vocab(dataset)
    return field


class InputDataset(Dataset):
    def __init__(self, texts: List[str], vocabulary: Field):
        examples = [
            Example.fromdict({"src": list(line)}, fields={"src": [("src", vocabulary)]})
            for line in texts
        ]
        super(InputDataset, self).__init__(
            examples=examples, fields=[("src", vocabulary)]
        )

    def get_iterator(self, batch_size=256, device="cuda"):
        return TorchIterator(self, batch_size=batch_size, device=device, train=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src)
        )
