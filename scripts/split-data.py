#!/usr/bin/env python

from itertools import groupby
from sys import argv

augments = ["white noise", "time stretch", "pitch scale", "random gain"]

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
  if len(argv) < 2:
    print("Usage: split-data.py <test_percent>")
    return

  test_percent = float(argv[1])

  with open("data/audio10.csv", "r") as f:
    header = f.readline()
    rows = [row.split(";") for row in f.readlines()]

  test = []
  train = []

  for _, g in groupby(rows, lambda row: row[-1]):
    rows = list(g)
    test_count = int(len(rows) * test_percent)

    test.extend(rows[:test_count])
    train.extend(rows[test_count:])

  add_augmented("data/audio10-aug-test.csv", header, test)
  add_augmented("data/audio10-aug-train.csv", header, train)

if __name__ == "__main__":
  main()
