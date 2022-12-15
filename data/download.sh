#!/bin/sh
set -euo pipefail

URI="https://accent.gmu.edu/soundtracks"
DEST="${1:-audio}"
PARALLEL="${2:-8}"

mkdir -p "$DEST"

curl -sL $URI | grep -E 'href="(.*)\.mp3"' | cut -d '"' -f 4 | tr -d "'" |
xargs -I{} -P "$PARALLEL" -- curl -s "$URI/{}" -o "$DEST/{}"
