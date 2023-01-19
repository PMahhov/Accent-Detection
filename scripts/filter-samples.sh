#!/bin/sh
set -euo pipefail

head -n 1 data/audio.csv >data/audio10.csv
tail -n +2 data/audio.csv | cut -d ';' -f 13 | uniq -c | awk '$1 >= 10 { print $2 }' >data/tmp

awk -F ';' 'NR == FNR { c[$1]++; next }; c[$13] > 0' data/tmp data/audio.csv >>data/audio10.csv
rm data/tmp
