# Basic Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Libraries for Classification and building Models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical

# Project Specific Libraries

import librosa
import librosa.display

df = pd.read_csv("urbansound8k/UrbanSound8K.csv")

'''We will extract classes from this metadata.'''

df.head()


def diagram_stuff():
    dat1, sampling_rate1 = librosa.load('urbansound8k/fold5/100032-3-0-0.wav')
    dat2, sampling_rate2 = librosa.load('urbansound8k/fold5/100263-2-0-117.wav')

    plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')

    plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')

    '''Using random samples to observe difference in waveforms.'''

    arr = np.array(df["slice_file_name"])
    fold = np.array(df["fold"])
    cla = np.array(df["class"])

    for i in range(192, 197, 2):
        path = 'urbansound8k/fold' + str(fold[i]) + '/' + arr[i]
        data, sampling_rate = librosa.load(path)
        plt.figure()  # figsize=(10, 5)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        plt.subplot(4, 2, 1)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(cla[i])
        plt.show()

'''EXAMPLE'''

dat1, sampling_rate1 = librosa.load('urbansound8k/fold5/100032-3-0-0.wav')
arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)


feature = []
label = []


def parser(row):
    # Function to load files and extract features
    for i in range(8732):
        file_name = 'urbansound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]

def ai_stuff():
    temp = parser(df)

    temp = np.array(temp)
    data = temp.transpose()

    X_ = data[:, 0]
    Y = data[:, 1]
    print(X_.shape, Y.shape)
    X = np.empty([8732, 128])

    for i in range(8732):
        X[i] = (X_[i])

    Y = to_categorical(Y)

    '''Final Data'''
    print(X.shape)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

    X_train = X_train.reshape(6549, 16, 8, 1)
    X_test = X_test.reshape(2183, 16, 8, 1)

    input_dim = (16, 8, 1)

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dim))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation="tanh"))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=90, batch_size=50, validation_data=(X_test, Y_test))

    model.summary()
    model.save('sound_identifier.h5')

    predictions = model.predict(X_test)
    score = model.evaluate(X_test, Y_test)
    print(score)

    predictions = np.argmax(predictions, axis=1)

    result = pd.DataFrame(predictions)
    result.to_csv("UrbanSound8kResults.csv")


# diagram_stuff()
ai_stuff()
