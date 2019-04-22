import glob
import os
import os.path


from typing import List, Tuple


def untokenize(sentence: List[str]) -> Tuple[str, str]:
    return "".join(sentence), " ".join(sentence)


def formatter(sequence):
    return "\t".join(untokenize(sequence)).replace("\n", "") + "\n"


def write_sentence(io_file, sentence: List[str], max_words=15):
    sequence = []
    for word in sentence:
        sequence.append(word)
        if len(sequence) == max_words:
            io_file.write(formatter(sequence))
            sequence = []

    if len(sequence):
        io_file.write(formatter(sequence))


def convert(input_path: str, output_path: str):
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
                input_fp.replace(input_path.split("**")[0], "")
            )
        )
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
        with open(input_fp) as input_fio:
            with open(output_fp, "w") as output_fio:

                sequence = []
                for line in input_fio.readlines():
                    tokens = line.split("\t")
                    sequence.append(tokens[0])

                    if line.startswith("."):
                        write_sentence(output_fio, sequence)
                        sequence = []

                # At the end of the loop, if sequence is not empty
                if sequence:
                    write_sentence(output_fio, sequence)


if __name__ == "__main__":
    convert("/home/thibault/dev/pie/datasets/jbc/**/*.tab", "data/")
