#!/bin/sh
set -euo pipefail

URI="https://accent.gmu.edu/browse_language.php?function=detail&speakerid="

for i in $(seq 1 3007); do
  for line in $(curl "$URI$i" | pup '.bio li json{}' | jq -r '.[]|(.children|.[]?.text),.text|select(.!=null)'); do

  done
done
