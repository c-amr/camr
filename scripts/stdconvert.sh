#!/bin/bash

#CORENLP_PATH='/home/j/llc/cwang24/Tools/CoreNLP-mod-convert.jar'
CORENLP_PATH='/home/j/llc/cwang24/Tools/CoreNLP-mod-convert-collapse.jar'
java -Xmx1800m -cp $CORENLP_PATH edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile $1 > $1.dep