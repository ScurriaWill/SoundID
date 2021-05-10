from keras.models import load_model
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical

model = load_model('sound_identifier')


def predict_digit(sound):
    sound = np.array(sound)
    # reshaping to support our model input
    sound = sound.reshape(16, 8, 1)
    # predicting the class
    res = model.predict([sound])[0]
    return np.argmax(res), max(res)


df = pd.read_csv("urbansound8k/UrbanSound8K.csv")

'''We will extract classes from this metadata.'''

df.head()

feature = []
label = []


def parser():
    # Function to load files and extract features
    for i in range(1):
        file_name = 'urbansound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # We extract mfcc feature from data
        place = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
        feature.append(place)
        label.append(df["classID"][i])
    return [feature, label]


temp = parser()

temp = np.array(temp)
data = temp.transpose()

x_ = data[:, 0]
y = data[:, 1]
print(x_.shape, y.shape)
x = np.empty([8732, 128])

for i in range(8732):
    x[i] = (x_[i])

y = to_categorical(y)

'''Final Data'''
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

x_train = x_train.reshape(6549, 16, 8, 1)
x_test = x_test.reshape(2183, 16, 8, 1)

input_dim = (16, 8, 1)
sound_type = str(predict_digit(x_test[1]))
print(sound_type)
