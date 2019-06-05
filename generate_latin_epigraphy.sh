#!/usr/bin/env bash

# Params :
# --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/edh_epidoc/gt test_data/edh_epidoc/txt/*.txt --max_char_length 150 --random_keep 0 --noise_char_random 0
venv/bin/python test_data/edh_epidoc/consolidate.py

#  --max_char_length 250
venv/bin/boudams dataset generate datasets/latin_epigraphy test_data/edh_epidoc/gt.tsv  --max_char_length 150
# venv/bin/python test_data/medieval_latin/consolidate.py
