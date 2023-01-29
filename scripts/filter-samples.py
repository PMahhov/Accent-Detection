#!/usr/bin/env python

from itertools import groupby
from pathlib import Path
from sys import argv

def with_suffix(path, suffix):
  p = Path(path)
  return str(p.with_stem(p.stem + suffix))

def reindex(rows, field_idx):
  unique = list(set(map(lambda row: int(row[field_idx]), rows)))
  unique.sort()
  dict = { str(f): str(i) for i, f in enumerate(unique) }

  for row in rows:
    row[field_idx] = dict[row[field_idx]]

def main():
  if len(argv) < 3:
    print("Usage: filter-samples.py <csv_file> <min_samples>")
    return

  csv_file = argv[1]
  min_samples = int(argv[2])

  with open(csv_file, "r") as f:
    header = f.readline().strip()
    header_cols = header.split(";")
    rows = [[col.strip() for col in row.split(";")] for row in f.readlines()]

  label_idx = header_cols.index("class_labels")
  primary_idx = header_cols.index("primary_lang_group_id")
  secondary_idx = header_cols.index("secondary_lang_group_id")

  filtered = []
  for _, g in groupby(rows, lambda row: row[label_idx]):
    rows = list(g)
    if len(rows) >= min_samples:
      filtered.extend(rows)

  with open(with_suffix(csv_file, str(min_samples)), "w") as new_csv:
    new_csv.write(header)

    reindex(filtered, label_idx)
    reindex(filtered, primary_idx)
    reindex(filtered, secondary_idx)

    for row in filtered:
      new_csv.write(";".join(row) + "\n")

if __name__ == "__main__":
  main()
