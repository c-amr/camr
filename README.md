Brandeis transition-based AMR Parser
==========

A Transition-based parser for [Abstract Meaning Representation](http://amr.isi.edu/).

# Dependencies
First download the project:
      
      git clone https://github.com/Juicechuan/AMRParsing.git

Here we use a modified version of the [Stanford CoreNLP python wrapper](https://github.com/dasmith/stanford-corenlp-python).
To setup Stanford CoreNLP, run the following scripts:
   
      cd stanfordnlp
      wget http://nlp.stanford.edu/software/stanford-corenlp-full-2013-06-20.zip
      unzip stanford-corenlp-full-2014-06-20.zip
      http://nlp.stanford.edu/software/stanford-parser-full-2014-01-04.zip	
      unzip stanford-parser-full-2014-01-04.zip

Put your splited data (current version only works for LDC2013E117) in data/

# Preprocesing
To preprocess the data, run:
   
      python amr_parsing.py -m preprocessing [input_amr_file]

This will give you the sentences(.sent), tokenized amr(.tok), POS tag and name entity (.prp) and dependency (.dep).
We use [JAMR](https://github.com/jflanigan/jamr) to get the alignment between sentence and its AMR annotation. You need to download and set up JAMR, then run the following script to get the aligned amr file:

      ./scripts/jamr_align.sh [input_amr_file]