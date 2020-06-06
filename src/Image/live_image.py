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
    agender = PyAgender()
    face = agender.detect_genders_ages(cv2.imread(path))
    gender = face[0]['gender']
    if gender >= 0.5:
        return 'Female'
    else:
        return 'Male'


def emotion_detect(path):
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
    cv2.imwrite('cv2_image.jpg', full_size_image)
    #     cv2.imshow('Emotion',full_size_image)
    #     cv2.waitKey(0)

    
def get_image_label(path):
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

def image_to_audio(path):
    label = get_image_label(path)
    fname = "../" + audio_df[audio_df['labels'] == label].sample(5)['path'].values[0]
    return fname
    

    
    
    
    
if __name__ == '__main__':
    image_to_audio('../../data/live_images/happy_test.jpg')

# from __future__ import division
# from keras.models import load_model
# import os
# import cv2
# import numpy as np
# from keras.preprocessing import image

# print('Loading Model...')
# model = load_model('../../data/fer2013.h5')
# model.load_weights('../../data/fer2013_weights.h5')
# print('Model Loaded.')

# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# cap=cv2.VideoCapture(0)

# while True:
#     ret,test_img=cap.read()# captures frame and returns boolean value and captured image
#     if not ret:
#         continue
#     gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


#     for (x,y,w,h) in faces_detected:
#         cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
#         roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
#         roi_gray=cv2.resize(roi_gray,(48,48))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis = 0)
#         img_pixels /= 255

#         predictions = model.predict(img_pixels)

#         #find max indexed array
#         max_index = np.argmax(predictions[0])

#         emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#         predicted_emotion = emotions[max_index]

#         cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#     resized_img = cv2.resize(test_img, (1000, 700))
#     cv2.imshow('Facial emotion analysis ',resized_img)



#     if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
#         break

# cap.release()
# cv2.destroyAllWindows