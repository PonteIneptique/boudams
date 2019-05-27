import glob
import os
import os.path
import random
import csv

from typing import Iterable, Union

from boudams.dataset.base import write_sentence


def convert(
        input_path: Union[Iterable[str], str], output_path: str, dict_reader: bool = True,
        min_words: int = 2, max_words: int = 10,
        min_char_length: int = 7, max_char_length: int = 100,
        random_keep: float = 0.3, max_kept: int = 1,
        noise_char: str = ".", noise_char_random: float = 0.2, max_noise_char: int = 2
):
    """ Build sequence to train data over using TSV or TAB files where either the first
    column or the column "form" or the column "token" contains the word that needs to be used.

    :param input_path: Glob-like path or list of path to treat
    :param output_path: Path where the file should be saved
    :param dict_reader: Files have column names
    :param min_words: Minimum of words to build a line
    :param max_words: Maximum number of words to build a line
    :param min_char_length: Minimum amount of characters to build a line
    :param max_char_length: Maximum amount of characters to build a line
    :param random_keep: Probability to keep some words for the next sequence
    :param max_kept: Maximum amount of words to be kept over next sequence
    :param noise_char: Character to add between words for noise purposes
    :param noise_char_random: Probability to add [NOISE_CHAR] in between words
    :param max_noise_char: Maximum amount of [NOISE_CHAR] to add sequentially
    """
    if isinstance(input_path, str):
        data = glob.glob(input_path)
    else:
        data = input_path

    for input_fp in data:
        output_fp = os.path.abspath(
            os.path.join(
                output_path,
                os.path.basename(input_fp)
            )
        )

        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
        key = "form"  # For dict reader

        with open(input_fp) as input_fio:
            with open(output_fp, "w") as output_fio:

                sequence = []
                next_sequence = random.randint(min_words, max_words)
                if dict_reader:
                    reader = csv.DictReader(input_fio, delimiter="\t", quotechar="漢")
                    if key not in reader.fieldnames:
                        key = "tokens"
                else:
                    reader = input_fio.readlines()

                for line_index, line in enumerate(reader):
                    if line_index == 0 and not dict_reader:
                        continue

                    if dict_reader:
                        sequence.append(line[key].strip())
                    else:
                        tokens = line.strip().split("\t")
                        sequence.append(tokens[0].strip())

                    char_length = len("".join(sequence))

                    # If the char length is greater than our maximum
                    #   we create a sentence now by saying next sequence is now.
                    if char_length > max_char_length * 0.9:
                        next_sequence = len(sequence)

                    # If we reached the random length for the word count
                    if len(sequence) == next_sequence:
                        # If however we have a string that is too small (like less then 7 chars), we'll pack it
                            # up next time
                        if char_length < min_char_length:
                            next_sequence += 1
                            continue

                        # If the sentence length is smaller than MAX_CHAR_LENGTH, we randomly add noise
                        if random.random() < noise_char_random:
                            index = random.randint(1, len(sequence))
                            sequence = sequence[:index] + \
                                       [noise_char] * random.randint(1, max_noise_char) + \
                                       sequence[index:]

                        write_sentence(output_fio, sequence)

                        # We randomly keep the last word for next sentence
                        if random.random() < random_keep:
                            kept = random.randint(1, max_kept)
                            sequence = sequence[-kept:] + []
                        else:
                            sequence = []

                        # We set-up the next sequence length
                        next_sequence = random.randint(min_words, max_words) + len(sequence)

                # At the end of the loop, if sequence is not empty
                if sequence and len("".join(sequence)) > min_char_length:
                    write_sentence(output_fio, sequence, max_chars=max_char_length)


if __name__ == "__main__":
    from boudams.dataset.base import check, split

    output = "/home/thibault/dev/boudams/data/seints"
    inp = "/home/thibault/dev/LiSeinConfessorPandora/data/lemmatises/*.tsv"

    convert(inp, output, dict_reader=True)
    split(output + "/*")
    check(output + "/")
