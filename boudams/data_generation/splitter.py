from typing import Iterable, List
from typing.re import Pattern
import regex as re
import random


def word_split(
        text: str,
        min_words: int = 2,
        max_words: int = 10,
        word_splitter: Pattern = r"(\s+)",
        **params
) -> Iterable[str]:
    word_splitter = re.compile(word_splitter)

    wbs = word_splitter.findall(text)
    words = word_splitter.split(text)

    sequences: List[List[str]] = [[]]
    next_sequence = random.randint(min_words, max_words)
    current_sequence = 0
    for word in words:
        sequences[-1].append(word)
        current_sequence += 1
        if current_sequence == next_sequence:
            sequences.append([])
            current_sequence = 0
            next_sequence = random.randint(min_words, max_words)
        else:
            sequences[-1].append(wbs.pop(0).replace("\n", " "))
    return [
        "".join(seq)
        for seq in sequences
    ]


def sentence_splitter(text: str, splitter: Pattern = r"(([\.\;!\?\"]+)") -> Iterable[str]:
    splitter = re.compile(splitter)
    wbs = splitter.findall(text)
    sentences = splitter.split(text)
    return [
        sent+wb
        for sent, wb in zip(sentences, wbs)
        if sent
    ]
