#!/bin/sh
set -euo pipefail

URI="https://accent.gmu.edu/browse_language.php?function=detail&speakerid="
DEST="${1:-html}"

echo "speaker_id;file_name;birth_place;native_langs;other_langs;age;sex;age_of_english_onset;english_learning_method;english_residence;years_of_english_residence"

for i in $(seq 1 3007); do
  echo -n "$i;"

  cat "$DEST/$i.html" |
  pup 'h5 em, .bio li json{}' |
  jq -r '.[]|(.children|.[]?.text),.text|select(.!=null and (startswith("(")|not))' |
  sed -r 's/(.*:$)/;/' |
  sed -r 's/([0-9]+), /\1;/' |
  sed -r 's/ years//' |
  tr -d '\n'

  echo
done
