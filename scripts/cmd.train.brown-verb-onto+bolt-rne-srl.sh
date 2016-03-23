#!/bin/sh

echo "Training Model ..."
/usr/bin/python -u amr_parsing.py -m train --amrfmt amr --prop --rne --onto 'onto+bolt' --smatcheval --model ./models/semeval/amr-semeval-all.train.basic-abt-brown-verb-onto+bolt-rne-srl -iter 5 --feat ./feature/basic_abt_srl_brown_feats.templates ./data/semeval/training.txt -d ./data/semeval/dev.txt > ./log/amr-semeval-all.train.basic-abt-brown-verb-onto+bolt-rne-srl.log 2>&1 &
