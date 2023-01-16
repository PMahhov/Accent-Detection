#!/usr/bin/python3

augments = ["white noise", "time stretch", "pitch scale", "random gain"]

def main():
  with open("data/audio10.csv", "r") as old_csv, open("data/audio10-augmented.csv", "w") as new_csv:
      header = old_csv.readline()
      new_csv.write(header)

      while row := old_csv.readline():
        new_csv.write(row)
        cols = row.split(";")
        file_name, sample_name = cols[1], cols[2]

        for augment in augments:
          cols[1] = f"{file_name} {augment}"
          cols[2] = f"{sample_name} {augment}"
          new_csv.write(";".join(cols))

if __name__ == "__main__":
  main()
