#!/bin/sh

echo "Training Model ..."
/usr/bin/python2.7 amr_parsing.py -m train --amrfmt amr --verblist --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb -iter 20 --feat ./feature/basic_abt_brown_feats.templates ./data/semeval/training.txt -d ./data/semeval/dev.txt > ./log/amr-semeval-all.train.basic-abt-brown-verb.log 2>&1 &
