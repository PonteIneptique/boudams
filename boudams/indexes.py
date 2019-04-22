from typing import List, Tuple
import torch


SOS_token = 0
EOS_token = 1


class Dictionary:
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        c = 0
        for word in sentence.split(' '):
            self.addUnit(word)
            c += 1
        return c

    def addUntokenized(self, sequence):
        for char in sequence:
            self.addUnit(char)
        return len(sequence)

    def addUnit(self, word: str):
        # Can be a word or a character
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __repr__(self):
        return "<Dictionary '"+self.name+"'>"

    @property
    def max_length(self):
        data = [len(x) for x in self.word2count]
        return max(data)

    @classmethod
    def load_dataset(cls, input_path: List[str]) -> Tuple["Dictionary", "Dictionary", int]:
        chars = cls("Tokenized")
        words = cls("NotTokenized")
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

                        length = chars.addUntokenized(inp)
                        if length > _max_lengths:
                            _max_lengths = length
                        length = words.addUntokenized(out)
                        if length > _max_lengths:
                            _max_lengths = length
        return chars, words, _max_lengths

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence]

    def tensorFromSentence(self, sentence, device):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


if __name__ == "__main__":
    import glob
    chars, words, max_length = Dictionary.load_dataset(glob.glob("data/**/*.tab"))
    print(chars.n_words, chars.word2count)
    print(words.n_words)
    print("Maximum length = " + str(max_length))
