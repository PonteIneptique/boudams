import os.path
import glob

for file in glob.glob("*.xml.txt"):
    with open(file) as inp, open(f"txt/{file}", "w") as out:
        content = []
        for line in inp.readlines():
            content.append(line.split("\t")[-1].strip())
        out.write(" ".join(content))