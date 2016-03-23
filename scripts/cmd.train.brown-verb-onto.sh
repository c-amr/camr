#!/bin/sh

echo "Training Model ..."
/usr/bin/python -u amr_parsing.py -m train --amrfmt amr --onto onto --verblist --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb-onto -iter 5 --feat ./feature/basic_abt_brown_feats.templates ./data/semeval/training.txt -d ./data/semeval/dev.txt > ./log/amr-semeval-all.train.basic-abt-brown-verb-onto.log 2>&1 &
