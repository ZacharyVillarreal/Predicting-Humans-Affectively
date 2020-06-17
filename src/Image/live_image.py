from __future__ import division
from keras.models import load_model
import os
import cv2
import numpy as np
import pandas as pd
from pyagender import PyAgender
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.playback import play


print('Loading Model...')
model = load_model('../../data/fer2013.h5')
model.load_weights('../../data/fer2013_weights.h5')
print('Model Loaded.')

# Image Resize
WIDTH = 48
HEIGHT = 48

audio_df = pd.read_csv("../../data/Emotional_audio_df.csv")
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def gender_detect(path):
    '''
    gender_detect: takes in an image file, uses OpenCV and 
    PyAgender to detected the gender of the individual in the 
    live image

    Parameters
    ----------
    file: file name + location where file is stored

    Returns
    -------
    Prediction string (i.e. "male")
    '''
    agender = PyAgender()
    face = agender.detect_genders_ages(cv2.imread(path))
    gender = face[0]['gender']
    if gender >= 0.5:
        return 'Female'
    else:
        return 'Male'

    
def get_image_label(path):
    '''
    get_image_label: takes in an image file, uses OpenCV for facial recognition,
    then it takes the facial image and contorts it to meet the input parameters of the 
    Convolutional NN, uses the loaded image CNN to make a prediction based off of the 
    contorted facial iamge

    Parameters
    ----------
    file: file name + location where file is stored

    Returns
    -------
    Prediction string (i.e. "male_angry")
    '''
    #loading image
    full_size_image = cv2.imread(path)
    print("Image Loaded")
    gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)

    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])
        print("Gender: " + gender_detect(path))
        
    label = gender_detect(path) + '_' + labels[int(np.argmax(yhat))]
    return label.lower()

def image_to_audio(file):
    '''
    image_to_audio: takes in an image file from app.py, runs the get_image_label function,
    then returns the predicted emotion and sex, then it queries into a pre-existing database 
    full of audio files that have matching labels, and will return the file name of the audio
    file with the matched label

    Parameters
    ----------
    file: file name + location where file is stored

    Returns
    -------
    Label: audio representation of image predicted emotion+sex (i.e. "male_angry.wav")
    '''
    label = get_image_label(file)
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

