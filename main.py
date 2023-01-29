import os
import sys
import tensorflow as tf
import pandas as pd
from keras.layers import Flatten, Dense,Dropout,MaxPooling2D, Conv2D
from keras.models import Sequential
import librosa
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt
from pydub import AudioSegment
from keras.callbacks import CSVLogger
from sklearn import metrics as mt
from keras_tuner.tuners import Hyperband

DOWN_CSV='audio10.csv'
CSV_TRAIN= 'audio10-aug-train.csv'
CSV_TEST= 'audio10-aug-test.csv'
CSV_VAL= 'audio10-aug-val.csv'
AUDIO_FILE = 'augmented'#'augmented'
AUDIO_FILE_IDEA = 'idea_audio_wav'
CSV_IDEA = 'idea_audio.csv'


SEED = 1337
EPOCHS = 100
RATE = 16000
def data_preprocessing( userPath):
    df = pd.read_csv(userPath , delimiter= ';', encoding = "ISO-8859-1", keep_default_na=False, na_values='')
    #replace space with comma delimeter
    df["other_langs"] = df["other_langs"].str.replace(" ",",")
    data = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return data
# Load the audio data and sample rate for each file
def get_Audio(data):
    userPath = os.getcwd()
    audio_data = []
    sample_rates = []
    for filename in data.file_name:
         filepath = os.path.join(userPath,'data',AUDIO_FILE,f'{filename}.wav')
         dataAud, sr = librosa.load(filepath,sr = RATE)
         audio_data.append(dataAud)
         sample_rates.append(sr)
    return audio_data,sample_rates
def get_MFCC(audio_data,sample_rates):
    mfccs = []
    for dataAud, sr in zip(audio_data, sample_rates):
        mfcc = librosa.feature.mfcc(y = dataAud,sr = RATE)
        mfccs.append(mfcc)
    return mfccs
def build_model(hp):
    num_classes = len(y_train[0])
    modelHyperTune = Sequential()
    modelHyperTune.add(Conv2D(filters=32,
                              kernel_size=(3, 3),
                              input_shape=(X_train.shape[1], X_train.shape[2], 1)))

    modelHyperTune.add(tf.keras.layers.BatchNormalization())
    modelHyperTune.add(Conv2D(filters=64,
                              kernel_size=(3, 3),
                              activation='relu'))

    modelHyperTune.add(tf.keras.layers.BatchNormalization())
    modelHyperTune.add(MaxPooling2D(pool_size=(2, 2)))
    modelHyperTune.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.3, step=0.1)))

    modelHyperTune.add(Flatten())

    modelHyperTune.add(Dense(hp.Int('dense_units_1', min_value=128, max_value=512, step=128, default=128),
        activation='relu'))
    modelHyperTune.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.3, step=0.1)))
    modelHyperTune.add(Dense(num_classes, activation='softmax'))

    hp_learning_rate = hp.Choice("learning_1", values=[1e-2, 1e-3, 1e-4, 1e-5])
    hp_weight_decay = hp.Choice("weight_1", values=[0.0, 1e-2, 1e-3, 1e-4, 1e-5])

    # Define optimizer, loss, and metrics
    modelHyperTune.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=hp_learning_rate, decay=hp_weight_decay),
        loss='categorical_crossentropy',
        metrics=["accuracy"])
    return modelHyperTune
def train_model(X_train,X_validation,y_train,y_validation):

    # Get row, column, and class sizes
    rows = X_train.shape[1]
    cols = X_train.shape[2]
    val_rows = X_validation.shape[1]
    val_cols = X_validation.shape[2]
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  # 32
                     data_format="channels_last",
                     input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    #Flatten array to 1-D
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    #output layer with the number of classes
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.000001, decay= 0.00001),
                  metrics=['accuracy'])

    #Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='val_loss', min_delta= .005, patience=5, verbose=1, mode='auto')
    #Write logs in CSV
    CSV_logger = tf.keras.callbacks.CSVLogger('model_specific_langs_group', separator=",", append=False)
    #Fit model
    history = model.fit(X_train, y_train,
                        steps_per_epoch= len(X_train)/ 32,
                        epochs=EPOCHS,
                        batch_size= 32,
                        callbacks=[es,CSV_logger],
                        validation_data=(X_validation, y_validation))

    return model,history
def hyperTune(X_train,X_test):
    tuner = Hyperband(build_model,
                         objective="val_accuracy",
                         max_epochs=50,
                         factor=3,
                         hyperband_iterations=10,
                         directory="kt_dir",
                         project_name="kt_hyperband")
    stop_early = EarlyStopping(monitor='val_loss', patience=5,min_delta=0.05)
    rows = X_train.shape[1]
    cols = X_train.shape[2]
    val_rows = X_validation.shape[1]
    val_cols = X_validation.shape[2]
    num_classes = len(y_train[0])

    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
    X_test = X_test.reshape(X_test.shape[0], val_rows, val_cols, 1)
    tuner.search(X_train, y_train, epochs=10, validation_split=0.3, callbacks=[stop_early], verbose=2)
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters()[0]

    # Build model
    h_model = tuner.hypermodel.build(best_hps)
    h_model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early], batch_size=32, verbose=2)

    return h_model
def plotGraph(model,history):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'], history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1, 2, 2)
    plt.plot(model.history.epoch, 100 * np.array(history.history['accuracy']),
             100 * np.array(history.history['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.show()

if __name__ == '__main__':
    #form user directory
    userPathTrain = os.path.join(os.getcwd(),'data',CSV_TRAIN)
    userPathTest = os.path.join(os.getcwd(),'data',CSV_TEST)
    userPathVal = os.path.join(os.getcwd(),'data',CSV_VAL)
    #userPathIDEA = os.path.join(os.getcwd(),'data',CSV_IDEA)

    #preprocess data and load the csv to form a dataframe
    data_train = data_preprocessing(userPathTrain)
    data_test = data_preprocessing( userPathTest)
    data_val = data_preprocessing( userPathVal)
    #data_IDEA = data_preprocessing( userPathIDEA)


    X_train = data_train.iloc[:, :-1]
    X_test = data_test.iloc[:, :-1]
    X_val = data_val.iloc[:, :-1]
    #X_test_IDEA = data_IDEA.iloc[:, :-2]

    y_train = data_train.iloc[:, -1]
    y_test = data_test.iloc[:, -1]
    y_val = data_val.iloc[:, -1]
   # y_test_IDEA =data_IDEA.iloc[:, -2]

    audio_data_TRAIN,sample_rates_TRAIN = get_Audio(X_train)
    audio_data_TEST,sample_rates_TEST = get_Audio(X_test)
    audio_data_VAL,sample_rates_VAL = get_Audio(X_val)
    #audio_data_IDEA, sample_rates_IDEA = get_Audio(X_test_IDEA)

    #get Mel-Spectogram Cepstral Coefficients
    data_MFCC_TRAIN = get_MFCC(audio_data_TRAIN,sample_rates_TRAIN )
    data_MFCC_TEST = get_MFCC(audio_data_TEST,sample_rates_TEST )
    data_MFCC_VAL = get_MFCC(audio_data_VAL,sample_rates_VAL )

    #data_MFCC_TEST_IDEAL = get_MFCC(audio_data_IDEA,sample_rates_IDEA )

    max_size_1 = max([elem.shape[1] for elem in data_MFCC_TRAIN])
    max_size_2 = max([elem.shape[1] for elem in data_MFCC_TEST])
    max_size_3 = max([elem.shape[1] for elem in data_MFCC_VAL])
    #max_size_4 = max([elem.shape[1] for elem in data_MFCC_TEST_IDEAL])
    #pad arrays with zeros
    max_size = max(max_size_1,max_size_2, max_size_3)#, max_size_4)
    data_MFCC_test = np.array(
        [np.pad(array, ((0, 0), (0, max_size - array.shape[1])), mode='constant') for array in data_MFCC_TEST])
    data_MFCC_train = np.array(
        [np.pad(array, ((0, 0), (0, max_size - array.shape[1])), mode='constant') for array in data_MFCC_TRAIN])
    data_MFCC_val = np.array(
        [np.pad(array, ((0, 0), (0, max_size - array.shape[1])), mode='constant') for array in data_MFCC_VAL])
    # data_MFCC_TEST_IDEA = np.array(
    #    [np.pad(array, ((0, 0), (0, max_size - array.shape[1])), mode='constant') for array in data_MFCC_TEST_IDEAL])


    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    #y_test_IDEA = to_categorical(y_test_IDEA)

    X_train = data_MFCC_train
    X_test = data_MFCC_test
    X_validation = data_MFCC_val
    #X_test_IDEA = data_MFCC_TEST_IDEAL

    model,history = train_model(X_train, X_validation, y_train, y_val)

    #plot accuracy and loss
    plotGraph(model,history)
    print(model.summary())
    '''
    HYPERPARAMETER TUNING
    h_model = hyperTune(X_train,X_validation)
    score = model.evaluate(X_test, y_test, verbose=3)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_pred = model.predict(X_test)

    # Convert predictions to a one-hot encoded format
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    conf_matrix = mt.confusion_matrix(y_test,y_pred)
    accuracy = mt.accuracy_score(y_test,y_pred)
    precision = mt.precision_score(y_test,y_pred)
    recall = mt.recall_score(y_test,y_pred)
    f1 = mt.f1_score(y_test,y_pred)
    print("Accuracy",precision)
    print("precision",precision)
    print("recall",recall)
    print("f1",f1)
    print("conf_matrix", conf_matrix)





