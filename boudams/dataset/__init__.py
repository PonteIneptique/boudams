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

    def tokenize(self, inp: str, out: str) -> Tuple[List[str], List[str]]:
        return list(inp), list(out)

    def __iter__(self) -> Iterator[Tuple[List[str], List[str]]]:
        # to be randomized later
        for file in self.files:
            with open(file) as f:
                for line in f.readlines():
                    data = line.strip()
                    if data:
                        yield self.tokenize(*tuple(data.split("\t")))

    def get_batches(self, n_seqs_in_a_batch, n_characters):
        '''Create a generator that returns batches of size
           n_seqs x n_steps from arr.

           Arguments
           ---------
           arr: Array you want to make batches from
           n_seqs: Batch size, the number of sequences per batch
           n_steps: Number of sequence steps per batch
        '''

        arr = list(self)
        batch_size = n_seqs_in_a_batch * n_characters
        n_batches = len(arr) // batch_size

        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size]
        # Reshape into n_seqs rows
        arr = arr.reshape((n_seqs_in_a_batch, -1))

        for n in range(0, arr.shape[1], n_characters):
            # The features
            x = arr[:, n:n + n_characters]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_characters]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

        yield x, y
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
