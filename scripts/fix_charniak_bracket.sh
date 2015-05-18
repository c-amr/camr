#!/bin/bash

cat $@ > tmp.txt

sed 's/^(S1 \((NP.*) (\. \.)\))$/(S1 (S \1))/g' tmp.txt > $@

rm tmp.txt
