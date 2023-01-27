#!/usr/bin/env python

from itertools import groupby
from sys import argv
from pathlib import Path

augments = ["white noise", "time stretch", "pitch scale", "random gain"]

def with_suffix(path, suffix):
  p = Path(path)
  return str(p.with_stem(p.stem + suffix))

def add_augmented(path, header, rows):
  with open(path, "w") as f:
    f.write(header)

    for row in rows:
      f.write(";".join(row))
      file_anme = row[1]
      sample_name = row[2]

      for aug in augments:
        row[1] = f"{file_anme} {aug}"
        row[2] = f"{sample_name} {aug}"
        f.write(";".join(row))

def main():
  if len(argv) < 3:
    print("Usage: split-data.py <csv_file> <test_percent>")
    return

  csv_file = argv[1]
  test_percent = float(argv[2])

  with open(csv_file, "r") as f:
    header = f.readline()
    rows = [row.split(";") for row in f.readlines()]

  train = []
  test = []
  val = []

  for _, g in groupby(rows, lambda row: row[-1]):
    rows = list(g)
    test_count = int(len(rows) * test_percent)

    test.extend(rows[:test_count])
    val.extend(rows[test_count:test_count * 2])
    train.extend(rows[test_count * 2:])

  add_augmented(with_suffix(csv_file, "-aug-train"), header, train)
  add_augmented(with_suffix(csv_file, "-aug-test"), header, test)
  add_augmented(with_suffix(csv_file, "-aug-val"), header, val)

if __name__ == "__main__":
  main()
