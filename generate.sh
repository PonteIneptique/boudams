#!/usr/bin/env bash

venv/bin/boudams dataset convert plain-text test_data/bfmmss/gt test_data/bfmmss/txt/*.txt
venv/bin/boudams dataset convert plain-text test_data/geste/gt test_data/geste/src/*.txt
venv/bin/boudams dataset convert plain-text test_data/bnf412/gt test_data/bnf412/txt/*.txt
venv/bin/boudams dataset convert plain-text test_data/himanis/gt/fro test_data/himanis/fro/*.txt
venv/bin/boudams dataset convert plain-text test_data/pinche/gt test_data/pinche/src/*.txt

venv/bin/boudams dataset generate datasets/fro test_data/bfmmss/gt/*.txt test_data/geste/gt/*.txt test_data/bnf412/gt/*.txt test_data/himanis/gt/fro/*.txt test_data/pinche/gt/*.txt