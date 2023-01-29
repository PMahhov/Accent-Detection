#!/Users/georgioschristopoulos/PycharmProjects/CNNAudio/venv/bin/python
# Reference:
# https://www.youtube.com/watch?v=umAXGVzVvwQ


import sys
import librosa as lb
import numpy as np

import random
from shutil import copyfile
import soundfile as sf
def white_noise(signal, noise_factor):
  return signal + noise_factor * np.random.normal(0, signal.std(), signal.size)

def time_stretch(signal, stretch_rate):
  return lb.effects.time_stretch(signal, rate = stretch_rate)

def pitch_scale(signal, sample_rate, num_semitones):
  return lb.effects.pitch_shift(signal, sr = sample_rate, n_steps = num_semitones)

def random_gain(signal, min_gain_factor, max_gain_factor):
  return signal * random.uniform(min_gain_factor, max_gain_factor)

def main():
  sample = sys.argv[1]
  copyfile(f"data/downsampled/{sample}.wav", f"data/augmented/{sample}.wav")
  signal, sample_rate = lb.load(f"data/downsampled/{sample}.wav")

  augmented = white_noise(signal, 0.2)
  sf.write(f"data/augmented/{sample} white noise.wav", augmented, sample_rate)

  augmented = time_stretch(signal, 1.25)
  sf.write(f"data/augmented/{sample} time stretch.wav", augmented, sample_rate)

  augmented = pitch_scale(signal, sample_rate, 4)
  sf.write(f"data/augmented/{sample} pitch scale.wav", augmented, sample_rate)

  augmented = random_gain(signal, 2, 4)
  sf.write(f"data/augmented/{sample} random gain.wav", augmented, sample_rate)

if __name__ == "__main__":
  main()
