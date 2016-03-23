#!/bin/sh

echo "Testing ..."
/usr/bin/python -u amr_parsing.py -m parse --amrfmt amr --onto onto --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb-onto.m ./data/semeval/test.txt > ./log/amr-semeval-all.test.test.basic-abt-brown-verb-onto.log 2>&1 &
