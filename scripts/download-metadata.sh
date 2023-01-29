#!/bin/sh
set -euo pipefail

URI="https://accent.gmu.edu/browse_language.php?function=detail&speakerid="
DEST="${1:-html}"
PARALLEL="${2:-8}"

mkdir -p "$DEST"

seq 1 3007 |
xargs -I{} -P "$PARALLEL" -- curl -s "$URI{}" -o "$DEST/{}.html"
