# Clean up files from the search fold of the source

import glob
import os

here = os.path.join(os.path.dirname(__file__))
sep = "******"


for file in glob.glob(os.path.join(here, "src", "*.txt")):
    with open(file) as f:
        text = f.read()
    text = text.split(sep)[-2]
    with open(file+".normalized.txt", "w") as f:
        f.write(text)
