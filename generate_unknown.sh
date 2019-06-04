#!/usr/bin/env bash

venv/bin/boudams dataset convert plain-text test_data/unknown/gt test_data/unknown/*.txt --min_words 2 --max_words 25 --min_char_length 8 --max_char_length 140 --random_keep 0 --noise_char_random 0
venv/bin/python test_data/unknown/consolidate.py