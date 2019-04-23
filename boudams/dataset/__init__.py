import random
from typing import List, Iterator, Tuple, Union
from torchtext.data import Field, ReversibleField, TabularDataset


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
CharacterField: ReversibleField = ReversibleField(tokenize=list, init_token="£", eos_token="$", lower=False)


def get_datasets(train, test, dev) -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
    fields = [("src", CharacterField), ("trg", CharacterField)]
    return (
        TabularDataset(train, format="TSV", fields=fields, skip_header=True),
        TabularDataset(test, format="TSV", fields=fields, skip_header=True),
        TabularDataset(dev, format="TSV", fields=fields, skip_header=True)
    )


def build_vocab(field: Field, datasets: Iterator[TabularDataset]) -> Field:
    for dataset in datasets:
        field.build_vocab(dataset)
    return field


def SOS_TOKEN(device="cpu", size=None):
    if size:
        inp = ([CharacterField.init_token], size)
    else:
        inp = [CharacterField.init_token]
    return CharacterField.numericalize(inp, device=device)
