#!/bin/sh

echo "Testing ..."
/usr/bin/python -u amr_parsing.py -m parse --amrfmt amr --prop --rne --onto --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb-onto-rne-srl-iter3.m ./data/semeval/test.txt > ./log/amr-semeval-all.test.basic-abt-brown-verb-onto-rne-srl.log 2>&1 &
