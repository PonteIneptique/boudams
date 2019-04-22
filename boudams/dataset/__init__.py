import random
from typing import List, Iterator, Tuple, Union


class Dataset:
    def __init__(self, path: Union[List[str], str]):
        if isinstance(path, str):
            path = [path]
        self.files = list(path)
        self.lengths = [[] for _ in path]
        self.iterable = 0
        for index, file in enumerate(path):
            with open(file) as fio:
                for line in fio.readlines():
                    if line.strip():
                        self.iterable += 1

    def tokenize(self, char_level, word_level):
        return list(char_level), list(word_level)#.split(" ")

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        # to be randomized later
        for file in self.files:
            with open(file) as f:
                for line in f.readlines():
                    data = line.strip()
                    if data:
                        yield self.tokenize(*tuple(data.split("\t")))

        #possibilities = (
        #    (file_index, sentence_index)
        #    for file_index, file_length in enumerate(self.lengths)
        #    for sentence_index in range(file_length)
        #)
        #for possibilities

if __name__ == "__main__":
    data = Dataset("data/test/test.tab")
    for inp, out in data:
        print(inp, out)
        break