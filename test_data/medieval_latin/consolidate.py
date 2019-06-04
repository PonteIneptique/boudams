import glob
import os

here = os.path.join(os.path.dirname(__file__))

with open(os.path.join(here, "..", "..", "datasets", "medieval_latin", "unknown.tsv"), "w") as output:
    for file in glob.glob(os.path.join(here, "monumenta", "gt", "*.txt")):
        with open(file) as input:
            for line in input.readlines():
                output.write(line)
