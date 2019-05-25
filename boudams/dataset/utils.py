import glob
import os
import os.path
import random
import re

from typing import List, Tuple


_space = re.compile("(\s+)")


def normalize_space(string: str) -> str:
    return _space.sub(" ", string)


def untokenize(sentence: List[str]) -> Tuple[str, str]:
    return "".join(sentence), " ".join(sentence)


def formatter(sequence):
    return "\t".join(untokenize(sequence)).replace("\n", "") + "\n"


def write_sentence(io_file, sentence: List[str], max_chars=150):
    sequence = []
    for word in sentence:
        if len(" ".join(sequence)) >= max_chars:
            io_file.write(formatter(sequence))
            sequence = []
        sequence.append(word)

    if len(sequence):
        io_file.write(formatter(sequence))


def convert(input_path: str, output_path: str, dict_reader: bool = True):
    """ Convert training files for PIE Ancient French to the model

    NONTOKENIZED\tTOKENIZED\n

    :param input_path:
    :param output_path:
    :return:
    """
    data = glob.glob(input_path)
    for input_fp in data:
        output_fp = os.path.abspath(
            os.path.join(
                output_path,
                os.path.basename(input_fp)
            )
        )
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
        with open(input_fp) as input_fio:
            with open(output_fp, "w") as output_fio:

                sequence = []
                MIN, MAX = 2, 10
                MIN_CHAR_LENGTH = 7
                MAX_CHAR_LENGTH = 100
                RANDOM_KEEP = 0.3
                NOISE_CHAR = "."
                NOISE_CHAR_RANDOM = 0.2
                MAX_NOISE_CHAR = 2
                MAX_KEPT = 1
                MAX_KEPT = 1

                next_sequence = random.randint(MIN, MAX)
                if dict_reader:
                    import csv
                    reader = csv.DictReader(input_fio, delimiter="\t", quotechar="漢")
                    key = "form"
                    if key not in reader.fieldnames:
                        key = "tokens"
                else:
                    reader = input_fio.readlines()

                for line_index, line in enumerate(reader):
                    if line_index == 0:
                        continue

                    if dict_reader:
                        sequence.append(line[key].strip())
                    else:
                        tokens = line.strip().split("\t")
                        sequence.append(tokens[0].strip())

                    char_length = len("".join(sequence))

                    # If the char length is greater than our maximum
                    #   we create a sentence now by saying next sequence is now.
                    if char_length > MAX_CHAR_LENGTH * 0.9:
                        next_sequence = len(sequence)

                    # If we reached the random length for the word count
                    if len(sequence) == next_sequence:
                        # If however we have a string that is too small (like less then 7 chars), we'll pack it
                            # up next time
                        if char_length < MIN_CHAR_LENGTH:
                            next_sequence += 1
                            continue

                        # If the sentence length is smaller than MAX_CHAR_LENGTH, we randomly add noise
                        if random.random() < NOISE_CHAR_RANDOM:
                            index = random.randint(1, len(sequence))
                            sequence = sequence[:index] + \
                                       [NOISE_CHAR] * random.randint(1, MAX_NOISE_CHAR) + \
                                       sequence[index:]

                        write_sentence(output_fio, sequence)

                        # We randomly keep the last word for next sentence
                        if random.random() < RANDOM_KEEP:
                            kept = random.randint(1, MAX_KEPT) #min(int(len(sequence) / 2), random.randint(1, MAX_KEPT))
                            sequence = sequence[-kept:] + []
                        else:
                            sequence = []

                        # We set-up the next sequence length
                        next_sequence = random.randint(MIN, MAX) + len(sequence)

                # At the end of the loop, if sequence is not empty
                if sequence and char_length > MIN_CHAR_LENGTH:
                    write_sentence(output_fio, sequence, max_chars=MAX_CHAR_LENGTH)


def split(input_path, ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    train_ratio, dev_ratio, test_ratio = ratio
    if train_ratio + dev_ratio + test_ratio != 1.0:
        raise AssertionError("Ratios sum should equal 1, got %s " % train_ratio + dev_ratio + test_ratio)

    directory = os.path.dirname(input_path)

    test_io = open(os.path.join(directory, "test.tsv"), "w")
    dev_io = open(os.path.join(directory, "dev.tsv"), "w")
    train_io = open(os.path.join(directory, "train.tsv"), "w")

    for file in glob.glob(input_path):
        if os.path.basename(file) in ("test.tsv", "dev.tsv", "train.tsv") or ".masked" in file:
            continue
        lines = []
        max_tr, max_te, max_de = 0, 0, 0
        with open(file) as read_io:
            lines = [index for index, _ in enumerate(read_io.readlines())]
            random.shuffle(lines)
            cut_tr = int(len(lines) * train_ratio)
            cut_de = cut_tr + int(len(lines) * dev_ratio)
            tr, de, te = lines[:cut_tr], lines[cut_tr:cut_de], lines[cut_de:]
            print(len(tr), len(de), len(te))

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

                if len(cur[0]) > 148 or len(cur[1].strip()) > 148:
                    print("TOO LARGE", line)
                    continue
                tgt.write(line)


def check(input_path, max_length=100):
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


if __name__ == "__main__":
    output = "/home/thibault/dev/boudams/data/seints"
    output = "/home/thibault/dev/boudams/data/fro"
    inp = "/home/thibault/dev/LiSeinConfessorPandora/data/lemmatises/*.tsv"
    inp = "/home/thibault/dev/boudams/data/inp/*.tab"
    convert(inp, output, dict_reader=True)
    split(output + "/*")
    check(output+"/")
