#!/bin/sh

echo "Training Model ..."
/usr/bin/python amr_parsing.py -m train --amrfmt amr --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt -iter 5 --feat ./feature/basic_abt_feats.templates ./data/semeval/training.txt -d ./data/semeval/dev.txt > ./log/amr-semeval-all.train.basic-abt.log 2>&1 &
