from keras.models import load_model
from time import sleep
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

img_model = load_model('data/fer2013.h5')
img_model.load_weights('data/fer2013_weights.h5')

audio_model = load_model('data/audio_nn.h5')
audio_model.load_weights('data/audio_nn_weights.h5')

for i in range(10):
    print('sleeping...')
    sleep(5)