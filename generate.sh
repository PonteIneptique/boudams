#!/usr/bin/env bash

# Params :
# --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/bfmmss/gt test_data/bfmmss/txt/*.txt --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/geste/gt test_data/geste/src/*.txt --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/bnf412/gt test_data/bnf412/txt/*.txt --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/himanis/gt/fro test_data/himanis/fro/*.txt --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/pinche/gt test_data/pinche/src/*.txt --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0

#  --max_char_length 250
venv/bin/boudams dataset generate datasets/bigger_fro test_data/bfmmss/gt/*.txt test_data/geste/gt/*.txt test_data/bnf412/gt/*.txt test_data/himanis/gt/fro/*.txt test_data/pinche/gt/*.txt  --max_char_length 250
