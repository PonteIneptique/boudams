import glob
import os

here = os.path.join(os.path.dirname(__file__))

with open(os.path.join(here, "..", "..", "datasets", "latin_epigraphy", "unknown.tsv"), "w") as output:
    with open(os.path.join(here, "..", "..", "datasets", "latin_epigraphy_uppercase", "unknown.tsv"), "w") as output_bis:
        for file in glob.glob(os.path.join(here, "pompei_gt", "*.txt")):
            with open(file) as input:
                for line in input.readlines():
                    output.write(line)
                    output_bis.write(line.upper())
