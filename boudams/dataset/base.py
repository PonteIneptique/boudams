import glob
import os
import random
import re
from typing import List, Tuple, Iterable, Union


_space = re.compile("(\s+)")


def normalize_space(string: str) -> str:
    """ Normalizes the space by replace any sequence of any spaces to a simple space ' ' (%20)"""
    return _space.sub(" ", string)


def untokenize(sentence: Iterable[str]) -> Tuple[str, str]:
    """ Transform a sequence of words into both a string without space (first
    element of the tuple) and a string with space (second element of the tuple)"""
    return "".join(sentence), " ".join(sentence)


def formatter(sequence: Iterable[str]):
    """ Joins a sequence of words into Training and Ground Truth format

    :param sequence: Sequence of words
    :return:
    """
    return "\t".join(untokenize(sequence)).replace("\n", "") + "\n"


def write_sentence(io_file, sentence: List[str], max_chars: int = 150):
    """ Write

    :param io_file: File to write to
    :param sentence: Sequence for training and ground_truth
    :param max_chars: Maximum number of characters to keep
    :return:
    """
    sequence = []
    for word in sentence:
        if len(" ".join(sequence)) >= max_chars:
            io_file.write(formatter(sequence))
            sequence = []
        sequence.append(word)

    if len(sequence):
        io_file.write(formatter(sequence))


def check(input_path: str, max_length: int = 100):
    """ Check train.tsv, dev.tsv and test.tsv in [INPUT_PATH] and print report

    :param input_path: Directory containing train, dev and test
    :param max_length: Maximum length of character for input or output
    """
    files = ("test.tsv", "dev.tsv", "train.tsv")
    for file in files:
        max_chars, min_chars, min_words, max_words = 0, max_length, max_length, 0
        with open(os.path.join(input_path, file)) as f:
            for line_index, line in enumerate(f.readlines()):
                if not line.strip():
                    continue
                x, y = tuple(line.strip().split("\t"))
                x_l = len(x)
                y_l = len(y)
                max_chars = max([max_chars, x_l, y_l])
                min_chars = min([min_chars, x_l, y_l])
                w = len(y.split())
                max_words = max(max_words, w)
                min_words = min(min_words, w)

                if max(x_l, y_l) > max_length:
                    print("ERROR: File %s : Line %s : %s " % (file, line_index, line))

                if x_l == 1 or y_l == 1:
                    print("ERROR: File %s : Line %s : %s " % (file, line_index, line))

            print("------")
            print("REPORT")
            print("File : " + file)
            print("Max char : %s" % max_chars)
            print("Min char : %s" % min_chars)
            print("Max words: %s" % max_words)
            print("Min words: %s" % min_words)
            print("------")


def split(input_path: Union[str, Iterable[str]], output_path: str, ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
          max_char_length: int = 150):
    """ Split a corpus of files into train, dev and test set

    :param input_path: List of path of Glib-Like path
    :param ratio: Ratio (Train, Dev, Test)
    :param max_char_length: Maximum length of input or output
    """
    max_char_length -= 2  # Remove SOS and EOS token

    train_ratio, dev_ratio, test_ratio = ratio
    if train_ratio + dev_ratio + test_ratio != 1.0:
        raise AssertionError("Ratios sum should equal 1, got %s " % (train_ratio + dev_ratio + test_ratio))

    if isinstance(input_path, str):
        input_path = glob.glob(input_path)

    test_io = open(os.path.join(output_path, "test.tsv"), "w")
    dev_io = open(os.path.join(output_path, "dev.tsv"), "w")
    train_io = open(os.path.join(output_path, "train.tsv"), "w")

    for file in input_path:
        if os.path.basename(file) in ("test.tsv", "dev.tsv", "train.tsv") or ".masked" in file:
            continue
        print("Treating %s" % file)
        lines = []
        max_tr, max_te, max_de = 0, 0, 0
        with open(file) as read_io:
            lines = [index for index, _ in enumerate(read_io.readlines())]
            random.shuffle(lines)
            cut_tr = int(len(lines) * train_ratio)
            cut_de = cut_tr + int(len(lines) * dev_ratio)
            tr, de, te = lines[:cut_tr], lines[cut_tr:cut_de], lines[cut_de:]
            print("-- %s line to train set" % len(tr))
            print("-- %s line to dev set" % len(de))
            print("-- %s line to test set" % len(te))

            read_io.seek(0)
            for line_index, line in enumerate(read_io.readlines()):
                if line_index in tr:
                    tgt = train_io
                    max_tr = max_tr
                elif line_index in te:
                    tgt = test_io
                else:
                    tgt = dev_io
                cur = line.split("\t")

                if len(cur[0]) > max_char_length or len(cur[1].strip()) > max_char_length:
                    print("---- [ERROR] Line %s is ignored because it's too large ! `%s`" % (line_index, line))
                    continue
                tgt.write(line)

    print("[DONE] Files available at %s " % output_path)
