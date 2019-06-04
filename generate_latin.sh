#!/usr/bin/env bash

# Params :
# --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/latin_prose/gt test_data/latin_prose/src/**/*.txt --max_char_length 150 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/latin_poetry/gt test_data/latin_poetry/src/*.txt --max_char_length 150 --random_keep 0 --noise_char_random 0

#  --max_char_length 250
venv/bin/boudams dataset generate datasets/latin test_data/latin_prose/gt/*.txt  --max_char_length 150
venv/bin/python test_data/latin_poetry/consolidate.py