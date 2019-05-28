import os

directory = os.path.dirname(__file__)

with open(os.path.join(directory, "txt", "guerin.txt")) as f:
	for line_index, line in enumerate(f.readlines()):

		if not line.strip():
			continue
			#
		lang, text = line[:3], line[7:].strip()
		with open(os.path.join(directory, lang, "guerin-%s.txt"% line_index), "w") as out_io:
			out_io.write(text)
