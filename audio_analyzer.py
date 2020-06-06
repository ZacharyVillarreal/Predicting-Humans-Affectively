from keras.models import load_model
from time import sleep
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pydub
import pickle
import pandas as pd 
import librosa
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cv2
from os import path
import sys
from pydub import AudioSegment


#Load Audio Recognition Neural Network

audio_model = load_model('data/audio_nn.h5')
audio_model.load_weights('data/audio_nn_weights.h5')

# Load Audio Scaler and Encoder

encoder = pickle.load(open('data/encoder.pickle', 'rb'))
scaler = pickle.load(open('data/scaler.pickle', 'rb'))


def separate(x):
    lst = []
    for i in x:
        lst.append(np.mean(i))
    return lst


# Audio Emotion Recognition


# def convert_to_m4a():
#     # Converts all audio to m4a format
#     folder = '../../data/live_audio/'
#     for filename in os.listdir(folder):
#         in_filename = os.path.join(folder, filename)
#         if not os.path.isfile(in_filename):
#             continue
#         oldbase = os.path.splittext(filename)
#         name = in_filename.replace('.tmp', '.m4a')
#         output = os.rename(in_filename, name)



# def convert_to_wav(file):
#     src = file
#     dst = "processed.wav"
#     sound = AudioSegment.from_m4a(src)
#     sound.export(dst, format = 'wav')
    
                    
def mfcc_live(file):
    df = pd.DataFrame(columns = ['mfcc_feature'])
    X, sample_rate = librosa.load(file, res_type = 'kaiser_fast',duration = 3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50)
    mfcc_scaled = np.mean(mfcc.T, axis = 0)
    df.loc[0] = [mfcc_scaled]
    mfcc_df = df.copy()
    mfcc_df.mfcc_feature = mfcc_df.mfcc_feature.apply(lambda x: separate(x))
    mfcc_list = ['mfcc'+str(i) for i in range(50)]
    
    df2 = pd.DataFrame(mfcc_df['mfcc_feature'].values.tolist(), columns = mfcc_list)
    return df2

def predict_live(filename):
    df = mfcc_live(filename)
    X = scaler.fit(np.array(df.iloc[:, :-1], dtype = float))
    X = scaler.transform(np.array(df.iloc[:, :-1], dtype = float))
    X = X.reshape(X.shape[0], X.shape[1],1)
    preds = audio_model.predict(X)
    print(preds)
    preds1=preds.argmax(axis=1)
    print(preds1)
    # predictions 
    preds = encoder.inverse_transform(preds1)
    blah = preds[0].split('_')
    print(blah[0])
    print(blah[1])
    return preds[0]

def audio_image(filename):
    label = predict_live(filename)
    if label == 'male_angry':
        return 'male_angry.jpg'
    if label == 'male_disgust':
        return 'male_disgust.jpg'
    if label == 'male_fear':
        return 'male_fear.jpg'
    if label == 'male_happy':
        return 'male_happy.jpg'
    if label == 'male_neutral':
        return 'male_neutral.jpg'
    if label == 'male_sad':
        return 'male_sad.jpg'
    if label == 'female_sad':
        return 'female_sad.jpg'
    if label == 'female_neutral':
        return 'female_neutral.jpg'
    if label == 'female_happy':
        return 'female_happy.jpg'
    if label == 'female_fear':
        return 'female_fear.jpg'
    if label == 'female_disgust':
        return 'female_disgust.jpg'
    if label == 'female_angry':
        return 'female_angry.jpg'


path = 'assets/' + sys.argv[-1]
print('We received this file: ', sys.argv[-1])
print(audio_image(path))