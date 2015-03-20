#!/bin/bash

# split data into train/dev/test folds

cat deft-amr-release-r3-proxy.txt | awk 'BEGIN {RS = ""} {gsub("\n", "\t", $0); print $0}' | grep '::preferred' | grep -v '_200[78]' | awk '{print $0; print ""}' | tr '\t' '\n' > deft-amr-release-r3-proxy.train
cat deft-amr-release-r3-proxy.txt | awk 'BEGIN {RS = ""} {gsub("\n", "\t", $0); print $0}' | grep '::preferred' | grep '_2007'       | awk '{print $0; print ""}' | tr '\t' '\n' > deft-amr-release-r3-proxy.dev
cat deft-amr-release-r3-proxy.txt | awk 'BEGIN {RS = ""} {gsub("\n", "\t", $0); print $0}' | grep '::preferred' | grep '_2008'       | awk '{print $0; print ""}' | tr '\t' '\n' > deft-amr-release-r3-proxy.test

