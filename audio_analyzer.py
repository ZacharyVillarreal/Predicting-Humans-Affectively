# from keras.models import load_model
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

audio_model = pickle.load(open('data/audio_model.pickle', 'rb'))

# Load Audio Scaler and Encoder

encoder = pickle.load(open('data/encoder.pickle', 'rb'))
scaler = pickle.load(open('data/scaler.pickle', 'rb'))


def separate(x):
    '''
    separate: separates the values within a list
    '''
    lst = []
    for i in x:
        lst.append(i)
    return lst


# Audio Emotion Recognition


    
                    
def mfcc_live(file):
    '''
    mfcc_live: takes in an audio file from app.py
    and performs feature extraction which specifically looks at 
    the mfcc of the audio file

    Parameters
    ----------
    file: file name + location where file is stored

    Returns
    -------
    DataFrame with mfcc values for inputted audio file
    '''
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

def predict_live(file):
    '''
    predict_live: takes in an audio file from app.py, runs the mfcc_live function
    above, converting it into its mfcc values, then runs it through the loaded
    CNN for audio files to predict both sex and emotion

    Parameters
    ----------
    file: file name + location where file is stored

    Returns
    -------
    Prediction string (i.e. "male_angry")
    '''
    df = mfcc_live(filename)
    X = scaler.transform(np.array(df.iloc[:, :-1], dtype = float))
    X = X.reshape(X.shape[0], X.shape[1],1)
    
    preds = audio_model.predict(X, batch_size = 16)
    preds = preds.argmax(axis=1)
    preds = preds.astype(int).flatten()
    
    # predictions 
    preds = encoder.inverse_transform(preds)[0]
    blah = preds.split('_')
    
    # Extracting sex and emotion separately for app.py
    print(blah[0])
    print(blah[1])
    return preds

def audio_image(file):
    '''
    audio_images: takes in an audio file from app.py, runs the mfcc_live function
    above, converting it into its mfcc values, then runs it through the loaded
    CNN for audio files to predict both sex and emotion in the predict_live function above,
    then queries into a database full of image representations matching the prediction and 
    returns the filepath for the prediction

    Parameters
    ----------
    file: file name + location where file is stored

    Returns
    -------
    Label: image representation of audio predicted emotion+sex (i.e. "male_angry.jpg")
    '''
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