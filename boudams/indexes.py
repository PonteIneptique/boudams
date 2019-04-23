from typing import List, Tuple
import torch


SOS_token = 0
EOS_token = 1


class Dictionary:
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, name):
        self.name = name
        self.chars2index = {}
        self.chars2count = {}
        self.index2chars = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def add_sentence(self, sequence) -> int:
        for char in sequence:
            self.add_char(char)
        return len(sequence)

    def add_char(self, word: str):
        # Can be a word or a character
        if word not in self.chars2index:
            self.chars2index[word] = self.n_chars
            self.chars2count[word] = 1
            self.index2chars[self.n_chars] = word
            self.n_chars += 1
        else:
            self.chars2count[word] += 1

    def __repr__(self):
        return "<Dictionary '"+self.name+"'>"

    @property
    def max_length(self):
        data = [len(x) for x in self.chars2count]
        return max(data)

    @classmethod
    def load_dataset(cls, input_path: List[str]) -> Tuple["Dictionary", int]:
        chars = cls("Tokenized")
        _max_lengths = 0
        for file in input_path:
            with open(file) as io:
                for line in io.readlines():
                    if line.strip() and "\t" in line:
                        try:
                            inp, out = tuple(line.strip().split("\t"))
                        except:
                            print(line)
                            raise

                        length = chars.add_sentence(inp)
                        if length > _max_lengths:
                            _max_lengths = length
                        length = chars.add_sentence(out)
                        if length > _max_lengths:
                            _max_lengths = length

        return chars, _max_lengths

    def get_indexes_from_sentence(self, sentence: str) -> List[int]:
        return [self.chars2index[word] for word in sentence]

    def get_tensor_from_sentence(self, sentence: str, device: str) -> torch.Tensor:
        indexes = self.get_indexes_from_sentence(sentence) + [EOS_token]
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


if __name__ == "__main__":
    import glob
    chars, max_length = Dictionary.load_dataset(glob.glob("data/**/*.tab"))
    print(chars.n_chars, chars.chars2count)
    print("Maximum length = " + str(max_length))
