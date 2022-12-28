#!/bin/sh
set -euo pipefail

SRC="${1:-html}"

printf "speaker_id;file_name;sample_name;birth_place;native_langs;other_langs;age;sex;age_of_english_onset;english_learning_method;english_residence;years_of_english_residence"

for i in $(seq 1 3007); do
  printf "\n$i;"

  cat "$SRC/$i.html" |
  pup 'source, h5 em, .bio li json{}' |
  jq '.[]|.src,(.children|.[]?.text),.text|select(.!=null and (startswith("(")|not))' |
  sed -r 's/.*:"$/;/' |
  sed -r 's/([0-9]+), /\1";"/' |
  sed -r 's/\/soundtracks\/(.+)\.mp3"/\1";/' |
  sed -r 's/ years//' |
  tr -d '\n'
done |
sed -r '/\/soundtracks\//d'
