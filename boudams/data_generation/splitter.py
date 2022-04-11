from typing import Iterable, List
import regex as re
import random


class Splitter:
    def split(self, text: str) -> Iterable[str]:
        raise NotImplementedError("BaseSplitter cannot be called.")


class WordSplitter(Splitter):
    def __init__(
            self,
            min_words: int = 2,
            max_words: int = 10,
            splitter: re.Pattern = r"\s+"
    ):
        self.min_words: int = min_words
        self.max_words: int = max_words
        self.splitter: re.Regex = re.compile(splitter)

    def split(self, text: str) -> Iterable[str]:
        wbs = self.splitter.findall(text)
        words = self.splitter.split(text)

        sequences: List[List[str]] = [[]]
        next_sequence = random.randint(self.min_words, self.max_words)
        current_sequence = 0
        for word in words:
            sequences[-1].append(word)
            current_sequence += 1
            if current_sequence == next_sequence:
                sequences.append([])
                current_sequence = 0
                next_sequence = random.randint(self.min_words, self.max_words)
                if wbs:
                    wbs.pop(0)
            elif wbs:
                sequences[-1].append(wbs.pop(0).replace("\n", " "))
        return [
            "".join(seq)
            for seq in sequences
        ]


class SentenceSplitter(Splitter):
    def __init__(self, splitter: re.Pattern = r"(([\.\;!\?\"]+)"):
        self.splitter: re.Regex = re.compile(splitter)

    def split(self, text: str) -> Iterable[str]:
        splitter = re.compile(self.splitter)
        wbs = splitter.findall(text)
        sentences = splitter.split(text)
        return [
            sent+wb
            for sent, wb in zip(sentences, wbs)
            if sent and sent.strip()
        ]
