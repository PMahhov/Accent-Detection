#!/usr/bin/env python

from itertools import groupby
from pathlib import Path
from sys import argv

def with_suffix(path, suffix):
  p = Path(path)
  return str(p.with_stem(p.stem + suffix))

def main():
  if len(argv) < 3:
    print("Usage: filter-samples.py <csv_file> <min_samples>")
    return

  csv_file = argv[1]
  min_samples = int(argv[2])

  with open(csv_file, "r") as f:
    header = f.readline()
    rows = [row.split(";") for row in f.readlines()]

  label_idx = header.split(";").index("class_labels")
  next_label = 0

  with open(with_suffix(csv_file, str(min_samples)), "w") as new_csv:
    new_csv.write(header)

    for _, g in groupby(rows, lambda row: row[label_idx]):
      rows = list(g)

      if len(rows) < min_samples:
        continue

      for row in rows:
        row[label_idx] = str(next_label)
        new_csv.write(";".join(row))

      next_label += 1

if __name__ == "__main__":
  main()
