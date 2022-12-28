#!/bin/sh
set -euo pipefail

URI="https://accent.gmu.edu/soundtracks/"
DEST="${1:-audio}"
PARALLEL="${2:-8}"

mkdir -p "$DEST"

curl -s $URI | pup 'tr td a attr{href}' |
xargs -I{} -P "$PARALLEL" -- curl -s "$URI{}" -o "$DEST/{}"
