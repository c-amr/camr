#!/bin/sh

echo "Testing ..."
/usr/bin/python -u amr_parsing.py -m parse --amrfmt amr --onto onto --rne --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb-onto-rne.m ./data/semeval/test.txt > ./log/amr-semeval-all.test.test.basic-abt-brown-verb-onto-rne.log 2>&1 &
