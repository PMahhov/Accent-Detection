#!/bin/sh

mkdir -p data/augmented

tail -n +2 data/audio10.csv | cut -d ';' -f 2 |
xargs -I{} -P $(nproc) -- ./scripts/augment-sample.py {}
