import sys
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
from pyagender import PyAgender

#Load Image Recognition Neural Network

img_model = load_model('data/fer2013.h5')
img_model.load_weights('data/fer2013_weights.h5')
# Image Emotion Recognition

# Image Resize
WIDTH = 48
HEIGHT = 48

audio_df = pd.read_csv("data/Emotional_audio_df.csv")
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def gender_detect(path):
    agender = PyAgender()
    face = agender.detect_genders_ages(cv2.imread(path))
    gender = face[0]['gender']
    if gender >= 0.5:
        return 'Female'
    else:
        return 'Male'


    
def get_image_label(path):
    #loading image
    full_size_image = cv2.imread(path)
#     print("Image Loaded")
    gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)
    yhat = 'test'

    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= img_model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print(str(labels[int(np.argmax(yhat))]))
        print(str(gender_detect(path)))
    
    label = gender_detect(path) + '_' + labels[int(np.argmax(yhat))]
    return label.lower()

def image_to_audio(path):
    label = get_image_label(path)
    if label == 'male_angry':
        return 'male_angry.wav'
    if label == 'male_disgust':
        return 'male_disgust.wav'
    if label == 'male_fear':
        return 'male_fear.wav'
    if label == 'male_happy':
        return 'male_happy.wav'
    if label == 'male_neutral':
        return 'male_neutral.wav'
    if label == 'male_sad':
        return 'male_sad.wav'
    if label == 'female_sad':
        return 'female_sad.wav'
    if label == 'female_neutral':
        return 'female_neutral.wav'
    if label == 'female_happy':
        return 'female_happy.wav'
    if label == 'female_fear':
        return 'female_fear.wav'
    if label == 'female_disgust':
        return 'female_disgust.wav'
    if label == 'female_angry':
        return 'female_angry.wav'
    



path = 'assets/' + sys.argv[-1]
print('We received this file: ', sys.argv[-1])
print(image_to_audio(path))