import os.path
import glob

for file in glob.glob("./gt/fro/*.txt"):
    with open(file) as inp, open(f"txt/{os.path.basename(file)}", "w") as out:
        content = []
        for line in inp.readlines():
            content.append(line.split("\t")[-1].strip())
        out.write(" ".join(content))
