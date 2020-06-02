import librosa
import librosa.display

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
%matplotlib inline
plt.style.use('ggplot')
import pandas as pd
import os
import IPython.display as ipd 
import pickle
import numpy
import csv
import scipy.misc
import scipy
import cv2

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

%load_ext autoreload
%autoreload 2

## Data import

df = pd.read_csv('../../data/mfcc_audio_df.csv')
df_x = df1.drop(['gender','emotion','source','path', 'true_label'], axis = 1)
y = df1.true_label

scaler = StandardScaler()
X = scaler.fit(np.array(df_x.iloc[:, :-1], dtype = float))
X = scaler.transform(np.array(df_x.iloc[:, :-1], dtype = float))
X = X.reshape(X.shape[0], X.shape[1],1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
y_train = np.array(y_train)
y_test = np.array(y_test)

# one hot encode the target (Categorical)
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))


# 1-D Convolutional model
def my_model():
    model = Sequential()
    model.add(Conv1D(64, 3, input_shape=(49, 1),activation='relu', padding='same'))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, 5, activation='relu',padding='same'))
#     model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256, 3,activation='relu',padding='same'))
#     model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=Adam(lr=0.001))
    
    return model


model=my_model()

# NN Stopping Mechanisms
earlystop = EarlyStopping(monitor='val_loss',
                          patience=20,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=20,
                              verbose=1,
                              min_delta=0.0001
                             )

# Fit Model
    
model.fit(x=X_train,     
                    y=y_train, 
                    batch_size=64, 
                    epochs=100, 
                    verbose=1, 
                    validation_data=(X_test,y_test),
                    callbacks=[earlystop, reduce_lr],
                    shuffle=True
            )

# Save Model
model.save('audio_nn.h5')
model.save_weights('audio_nn_weights.h5')
pickle.dump(lb, open('../../data/Saved_Files/encoder.p', 'wb'))
pickle.dump(scaler, open('../../data/Saved_Files/scaler.p', 'wb'))