#!/usr/bin/env bash

# Params :
# --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/medieval_latin/monumenta/gt test_data/medieval_latin/monumenta/src/*.txt --max_char_length 150 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/medieval_latin/cartae/gt test_data/medieval_latin/cartae/src/*.normalized.txt --max_char_length 150 --random_keep 0 --noise_char_random 0

#  --max_char_length 250
venv/bin/boudams dataset generate datasets/medieval_latin test_data/medieval_latin/cartae/gt/*.txt  --max_char_length 150
venv/bin/python test_data/medieval_latin/consolidate.py
