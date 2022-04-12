import glob
import os
import os.path
import random
import regex as re

from typing import Iterable, Union, Dict

from boudams.modes import SimpleSpaceMode
from boudams.data_generation.splitter import Splitter


_SPACES = re.compile(r"(\s+)")


def convert(
    input_path: Union[Iterable[str], str],
    output_path: str,
    splitter: Splitter,
    token_ratio: Dict[str, float] = None,
    mode: SimpleSpaceMode = None,
    min_chars: int = 5,
    **kwargs
):
    """ Build sequence to train data over using TSV or TAB files where either the first
    column or the column "form" or the column "token" contains the word that needs to be used.

    :param input_path: Glob-like path or list of path to treat
    :param output_path: Path where the file should be saved
    :param splitter:
    :param token_ratio:
    :param mode:
    """

    # ToDo: Reimplement noise maker, probably with some class ?

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

        with open(input_fp) as input_fio, open(output_fp, "w") as output_fio:
            for line in input_fio.readlines():
                if mode.NormalizeSpace:
                    line = _SPACES.sub(" ", line)
                for sequence in splitter.split(line.strip()):
                    if sequence.strip():
                        sample, mask = mode.generate_mask(sequence, tokens_ratio=token_ratio)
                        if len(sample) >= min_chars:
                            output_fio.write("\t".join([sample, mask])+"\n")


if __name__ == "__main__":
    from boudams.data_generation.base import check, split
    from boudams.data_generation.splitter import WordSplitter

    inp = "/home/thibault/dev/boudams/test_data/*/txt/*.txt"
    output = "/home/thibault/dev/boudams/test_data/new"

    max_char_length = 200

    convert(inp, output, splitter=WordSplitter(min_words=5, max_words=20), mode=SimpleSpaceMode())
    split(output + "/*.txt", output_path=output)
    #check(output + "/")
