import os
import sys
import pandas as pd
from keras.layers import Flatten, Dense,Dropout,MaxPooling2D, Conv2D
from keras.models import Sequential
import librosa
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append('/Users/georgioschristopoulos/Desktop/ffmpeg')

SEED = 1337
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_RATIO = 0.1
COL_SIZE = 30
RATE = 24000
N_MFCC = 13
# Set the input shape for the audio data
input_shape = (None, 1)
# Load the MP3 file
def data_preprocessing(userPath):
    df = pd.read_csv(userPath, on_bad_lines = 'skip' , delimiter= ';', encoding = "ISO-8859-1", keep_default_na=False, na_values='')
    #replace space with comma delimeter
    df["other_langs"] = df["other_langs"].str.replace(" ",",")
    data = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return data
# Load the audio data and sample rate for each file
def get_Audio(data):
    userPath = os.getcwd()
    #filenames = librosa.util.find_files(userPath + "/downsampled_wav")
    audio_data = []
    sample_rates = []
    for filename in data.file_name:
         filepath = os.path.join(userPath,'data','downsampled',f'{filename}.wav')
         #audio_file_name = filename.rsplit("/",1)[1].split(".",1)[0]
         #sound = AudioSegment.from_mp3(filename)
         #sound.export("/Users/georgioschristopoulos/PycharmProjects/CNNAudio/data/wav_files/"+audio_file_name+".wav", format="wav")
         dataAud, sr = librosa.load(filepath)
    # dataAud = dataAud.T
    # downsampled_wav = librosa.resample(dataAud, orig_sr=sr, target_sr=RATE, scale=True)
     #dataAud = dataAud.T
         audio_data.append(dataAud)
         sample_rates.append(sr)
         #sf.write(os.path.join(userPath+'/downsampled_wav/', audio_file_name+".wav"), downsampled_wav, RATE)
    return audio_data,sample_rates

def get_MFCC(audio_data,sample_rates):
    mfccs = []
    for dataAud, sr in zip(audio_data, sample_rates):
        mfcc = librosa.feature.mfcc(y = dataAud,sr = RATE)
        mfccs.append(mfcc)
    return mfccs


def remove_silence(wav, thresh=0.04, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(len(wav) / chunk):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])

def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs.native_langs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


def train_model(X_train,X_validation,y_train,y_validation, batch_size=128): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    # Get row, column, and class sizes
    rows = X_train.shape[1]
    cols = X_train.shape[2]
    val_rows = X_validation.shape[1]
    val_cols = X_validation.shape[2]
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)
    #10,20,30


    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, input_shape=(X_train.shape[0],),activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='accuracy', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)
    print(len(X_train))
    print(len(y_train))
    print(len(X_validation))
    print(len(y_validation))
    # Fit model using ImageDataGenerator
    model.fit(X_train, y_train,
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS,
                        callbacks=[es,tb], validation_data=(X_validation,y_validation))

    return (model)

if __name__ == '__main__':
    #form user directory
    userPath = os.path.join(os.getcwd(),'data','audio.csv')
    #preprocess data and load the csv to form a dataframe
    data = data_preprocessing(userPath)
    #for each language create the frequencies
    temp = data.class_labels.value_counts()
    #split data
    X_train, X_test, y_train, y_test =train_test_split(data.iloc[:,:-1], data["class_labels"], test_size=0.2,random_state=42)

    #load downsampled audio files from directory
    audio_data,sample_rates = get_Audio(X_train)
    #get the MFCC for the training sample
    data_MFCC_train = get_MFCC(audio_data,sample_rates )

    # load downsampled audio files from directory
    audio_data,sample_rates = get_Audio(X_test)

    # get the MFCC for the training sample
    data_MFCC_test = get_MFCC(audio_data,sample_rates )

    max_size_1= max([elem.shape[1] for elem in data_MFCC_train])
    max_size_2= max([elem.shape[1] for elem in data_MFCC_test])
    max_size  = max(max_size_1, max_size_2)
    data_MFCC_test = np.array([np.pad(array, ((0,0), (0, max_size - array.shape[1])), mode='constant') for array in data_MFCC_test])
    data_MFCC_train = np.array([np.pad(array, ((0,0), (0, max_size - array.shape[1])), mode='constant') for array in data_MFCC_train])

    #transofrm values to binary
    y_train = to_categorical(y_train ,num_classes=data.class_labels.unique().size)
    y_test = to_categorical(y_test, num_classes=data.class_labels.unique().size)
    model = train_model(data_MFCC_train, data_MFCC_test,y_train, y_test)

    #y_pred = model.predict(np.asarray(X_test))

    print("finished")





