import os
import sys
import argparse
from pydub import AudioSegment
from feature_extraction import *
import librosa
import librosa.display
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


from keras.models import load_model
model = load_model('../../data/audio_nn.h5')
model.load_weights('../../data/audio_nn_weights.h5')


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

    for (dirpath, dirnames, filenames) in os.walk("../../data/live_audio/"):
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
    print('CONVERTING DONE.')
    
    
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
#     df2['path'] = path
    return df2

def predict_live(filename):
    df = mfcc_live(filename)
    scaler = StandardScaler()
    X = scaler.fit(np.array(df.iloc[:, :-1], dtype = float))
    X = scaler.transform(np.array(df.iloc[:, :-1], dtype = float))
    X = X.reshape(X.shape[0], X.shape[1],1)
    preds = model.predict(X, 
                         batch_size=16, 
                         verbose=1)

    preds1=preds.argmax(axis=1)
    # predictions 
#     predictions = (LabelEncoder.inverse_transform((preds1)))[0]
    preds = pd.DataFrame({'predicted': preds1})
    return preds

