#!/bin/bash

echo "Setup Stanford CoreNLP ..."
cd stanfordnlp
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip

echo "Setup Charniak Parser ..."
pip install --user bllipparser
