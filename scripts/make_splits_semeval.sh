#!/bin/bash
set -ueo pipefail

# split amr_anno_1.0 split data into train/dev/test folds

#DATA_DIR="${JAMR_HOME}/data/amr_anno_1.0/data/split"
DATA_DIR="$1/data/amrs/split"

split=training
cat "${DATA_DIR}/$split/deft-p2-amr-r1-amrs-$split-proxy.txt" > "${DATA_DIR}/$split/$split.txt"
for corpus in bolt cctv dfa guidelines mt09sdl wb xinhua; do
    cat "${DATA_DIR}/$split/deft-p2-amr-r1-amrs-$split-$corpus.txt" >> "${DATA_DIR}/$split/$split.txt"
done

split=dev
cat "${DATA_DIR}/$split/deft-p2-amr-r1-amrs-$split-proxy.txt" > "${DATA_DIR}/$split/$split.txt"
for corpus in bolt consensus dfa xinhua; do
    cat "${DATA_DIR}/$split/deft-p2-amr-r1-amrs-$split-$corpus.txt" >> "${DATA_DIR}/$split/$split.txt"
done

split=test
cat "${DATA_DIR}/$split/deft-p2-amr-r1-amrs-$split-proxy.txt" > "${DATA_DIR}/$split/$split.txt"
for corpus in bolt consensus dfa xinhua; do
    cat "${DATA_DIR}/$split/deft-p2-amr-r1-amrs-$split-$corpus.txt" >> "${DATA_DIR}/$split/$split.txt"
done
