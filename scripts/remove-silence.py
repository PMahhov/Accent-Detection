#!/usr/bin/python3

from os import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
import scipy.io.wavfile as wav
import numpy as np

def remove_silence(filename):
  print(f"Processing {filename}")
  rate, data = wav.read(f"data/downsampled/{filename}")

  audio = AudioSegment(data.tobytes(), frame_rate = rate, sample_width = data.dtype.itemsize, channels = 1)
  audio_chunks = split_on_silence(audio, min_silence_len = 200, silence_thresh = -50, keep_silence = 20)

  processed = np.array(sum(audio_chunks).get_array_of_samples())
  wav.write(f"data/no-silence/{filename}", rate, processed)

def main():
  if len(sys.argv) < 2:
    print("Usage: remove-silence.py <filename>")
    return

  remove_silence(sys.argv[1])

if __name__ == "__main__":
  main()
