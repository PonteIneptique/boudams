#!/usr/bin/env bash

# Params :
# --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 250 --random_keep 0 --noise_char_random 0
venv/bin/boudams dataset convert plain-text test_data/edh_epidoc/pompei_gt test_data/edh_epidoc/pompei/*.txt --max_char_length 150 --random_keep 0 --noise_char_random 0
venv/bin/python test_data/edh_epidoc/consolidate_pompei.py
