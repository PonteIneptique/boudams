# Transform to conlu the datasets in fro ?

import glob
for filepath in glob.glob("../datasets/**/*.tsv"):
    with open(filepath) as f:
        with open(filepath.replace(".tsv", ".conl"), "w") as io:
            for line in f.readlines():
                if "\t" not in line:
                    continue
                inp, out = line.strip().split("\t")
                for word_index, word in enumerate(out.split()):
                    io.write("{}\t{}\t".format(word_index+1, word)+"\t".join(["_"]*7+["SpaceAfter=No"])+"\n")
                io.write("\n")

