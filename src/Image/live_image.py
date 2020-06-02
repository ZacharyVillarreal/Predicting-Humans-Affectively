from __future__ import division
from keras.models import load_model
import os
import cv2
import numpy as np


print('Loading Model...')
model = load_model('../../data/fer2013.h5')
model.load_weights('../../data/fer2013_weights.h5')
print('Model Loaded.')

# Image Resize
WIDTH = 48
HEIGHT = 48

labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

#loading image
full_size_image = cv2.imread('/home/ubuntu/Emotional-Training/data/live_images/happy_test.jpg')
print("Image Loaded")
gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
face = cv2.CascadeClassifier(cv2.data.haarcascades + '/home/ubuntu/Emotional-Training/src/haarcascade_frontalface_default.xml')
# faces = face.detectMultiScale(gray, 1.3, 10)

#detecting faces
# for (x, y, w, h) in faces:
#     roi_gray = gray[y:y + h, x:x + w]
#     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#     cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
#     cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
#     #predicting the emotion
#     yhat= model.predict(cropped_img)
#     cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
#     print("Emotion: "+labels[int(np.argmax(yhat))])


cv2.imshow('Emotion',full_size_image)

cv2.waitKey()