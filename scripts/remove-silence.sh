#!/bin/sh
set -euo pipefail

mkdir -p data/no-silence

ls data/downsampled |
xargs -I{} -P 16 -- ./scripts/remove-silence.py {}
