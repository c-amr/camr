#!/bin/sh

echo "Training Model ..."
/usr/bin/python amr_parsing.py -m train --amrfmt amr --rne --onto --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb-onto-rne -iter 5 --feat ./feature/basic_abt_brown_feats.templates ./data/semeval/training.txt -d ./data/semeval/dev.txt > ./log/amr-semeval-all.train.basic-abt-brown-verb-onto-rne.log 2>&1 &
