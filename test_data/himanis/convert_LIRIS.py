import os

directory = os.path.dirname(__file__)
lang = "lat"

with open(os.path.join(directory, "txt", "LIRIS_mss_dates_w.txt")) as f:
	for line_index, line in enumerate(f.readlines()):
		if not line.strip():
			continue
			#
		with open(os.path.join(directory, lang, "LIRIS-%s.txt"% line_index), "w") as out_io:
			out_io.write(line.strip())
