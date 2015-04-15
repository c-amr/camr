#!/bin/bash

#JAMR_HOME="/home/j/llc/cwang24/Tools/jamr"

#### Config ####
${JAMR_HOME}/scripts/config.sh

#### Align the tokenized amr file ####

echo "### Aligning $1 ###"

${JAMR_HOME}/run Aligner -v 0 < $1.tok > $1.aligned