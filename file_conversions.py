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


def convert_to_m4a():
    # Converts all audio to m4a format
    folder = '../../data/live_audio/'
    for filename in os.listdir(folder):
        in_filename = os.path.join(folder, filename)
        if not os.path.isfile(in_filename):
            continue
        oldbase = os.path.splittext(filename)
        name = in_filename.replace('.tmp', '.m4a')
        output = os.rename(in_filename, name)



def convert_to_wav():
    formats_to_convert = ['.m4a']

    for (dirpath, dirnames, filenames) in os.walk("data/live_audio"):
        for filename in filenames:
            if filename.endswith(tuple(formats_to_convert)):

                filepath = dirpath + '/' + filename
                (path, file_extension) = os.path.splitext(filepath)
                file_extension_final = file_extension.replace('.', '')
                try:
                    track = AudioSegment.from_file(filepath,
                            file_extension_final)
                    wav_filename = filename.replace(file_extension_final, 'wav')
                    wav_path = dirpath + '/' + wav_filename
                    print('CONVERTING: ' + str(filepath) + '...')
                    file_handle = track.export(wav_path, format='wav')
                    os.remove(filepath)
                except:
                    print("ERROR CONVERTING " + str(filepath))
                    
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
    scaler = StandardScaler()
    X = scaler.fit(np.array(df.iloc[:, :-1], dtype = float))
    X = scaler.transform(np.array(df.iloc[:, :-1], dtype = float))
    X = X.reshape(X.shape[0], X.shape[1],1)
    preds = audio_model.predict(X, 
                         batch_size=16, 
                         verbose=1)

    preds1=preds.argmax(axis=1)
    # predictions 
    preds = encoder.inverse_transform(preds1)
    print(preds[0])
    return preds[0]

def audio_image(filename):
    label = predict_live(filename)
    try:
        img = Image.open("data/live_images/" + label + ".jpg")
        img.save("assets/response.jpg")
        print('Saved')
    except IOError:
        print('Error')
        
        
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
        yhat= img_model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])
        print("Gender: " + gender_detect(path))
        
        label = gender_detect(path) + '_' + labels[int(np.argmax(yhat))]
        
    return label.lower()

def image_to_audio(path):
    label = get_image_label(path)
    blah = label.split('_')
    print('Gender: ', blah[0])
    print('Emotion: ', blah[1])
    fname = "../" + audio_df[audio_df['labels'] == label].sample(5)['path'].values[0]
    return fname

