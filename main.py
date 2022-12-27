import hashlib
import os
import pandas as pd
import tensorflow
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import librosa
import soundfile as sf
# Find all the audio files in the 'audio' folder

# Set the input shape for the audio data
input_shape = (None, 1)
userPath = os.getcwd() + '/data'
# Load the MP3 file

filenames = librosa.util.find_files(userPath + "/audio")
df = pd.read_csv(userPath +"/audio.csv", on_bad_lines = 'skip' , delimiter= ';')
# Load the audio data and sample rate for each file
audio_data = []
sample_rates = []
for filename in filenames:
    data, sr = sf.read(filename)
    data = data.T
    audio_data.append(data)
    sample_rates.append(sr)

mfccs = []
for data, sr in zip(audio_data, sample_rates):
    mfcc = librosa.feature.mfcc(y = data,sr = sr)
    mfccs.append(mfcc)

# Combine the features and labels into a single dataset
X = mfccs
y = df["native_language"]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the model
model = Sequential()

# Add a 1D convolutional layer with 32 filters of size 3
model.add(Conv1D(32, 3, input_shape=input_shape, activation='relu'))

# Add a max pooling layer with a pool size of 2
model.add(MaxPooling1D(pool_size=2))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Add a dense layer with 64 units
model.add(Dense(64, activation='relu'))

# Add the final output layer with a single unit
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimization
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the audio data and labels
X = np.load('audio_data.npy')
y = np.load('audio_labels.npy')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

if __name__ == '__main__':
    print("hellooo")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/




