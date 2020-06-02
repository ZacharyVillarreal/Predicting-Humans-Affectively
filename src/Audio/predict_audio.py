import librosa
import librosa.display

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import pandas as pd
import os
import IPython.display as ipd 
import pickle
import numpy
import csv
import scipy.misc
import scipy
import cv2

from feature_extraction import *

#Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Neural Network Imports
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from clean_audio_data import *


def prediction(filename, model_name):
    
    from keras.models import load_model
    model = load_model('../../data/audio_nn.h5')
    model.load_weights('../../data/audio_nn_weights.h5')

#     encoder = pickle.load(open('../../data/Saved_Files/encoder.p', 'rb'))
#     scaler = pickle.load(open('../../data/Saved_Files/scaler.p', 'rb'))
    
    df = mfcc(filename)
    X = StandardScaler.transform(np.array(df))
    X = X.reshape(X.shape[0], X.shape[1],1)
    
    # predictions 
    preds = preds.astype(int).flatten()
    preds = (LabelEncoder.inverse_transform((preds)))
    preds = pd.DataFrame({'predicted': preds})

    # Actual labels
    actual=y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (LabelEncoder.inverse_transform((actual)))
    actual = pd.DataFrame({'actual': actual})

    # Lets combined both of them into a single dataframe
    finaldf = actual.join(preds)

    return finaldf


def accuracy(df):
    accuracy = df[df['correct'] == True].count()[1]/df.groupby('correct', as_index=False).count().sum()[1]
    return 'Accuracy = ' + str(round(accuracy*100,2)) + '%'



def pred_df(model): 
    paths = []
    for (dirpath, dirnames, filenames) in os.walk('live_audio'):
        for filename in filenames:
            if filename.endswith('.wav'): 
                paths.append(os.sep.join([dirpath, filename]))
            
            
    df = pd.DataFrame(columns=['name', 'actual', 'predicted'])
    for filename in paths:
        preds = prediction(filename, model)
        df = df.append(preds, ignore_index=True)
        df['correct'] = np.where(df['predicted'] == df['actual'],True, False)

    print(accuracy(df))
    
    return df


def pred_df_mf(model, people_dict): 
    paths = []
    for (dirpath, dirnames, filenames) in os.walk('live_audio'):
        for filename in filenames:
            if filename.endswith('.wav'): 
                paths.append(os.sep.join([dirpath, filename]))
            
            
    df = pd.DataFrame(columns=['name', 'actual', 'predicted'])
    for filename in paths:
        preds = prediction(filename, model)
        df = df.append(preds, ignore_index=True)

    df['gender'] = df['name'].apply(lambda x: people_dict[x])
    df['actual'] = df['gender'] + '_' + df['actual']
    df['correct'] = np.where(df['predicted'] == df['actual'],True, False)

    print(accuracy(df))
    
    return df
    
