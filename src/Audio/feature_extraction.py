import librosa
import librosa.display
import numpy as np
import seaborn as sns
import pandas as pd

import pandas as pd
import numpy as np
import os
import librosa

def separate(x):
    '''
    separate: separates the values within a list and returns the mean
    '''
    lst = []
    for i in x:
        lst.append(np.mean(i))
    return lst


def mfcc(col):
    '''
    mfcc_live: takes in an audio file from the file path column in 
    the audio dataframe and performs feature extraction which 
    specifically looks at the mfcc of that audio file

    Parameters
    ----------
    col: pandas dataframe colun

    Returns
    -------
    DataFrame with mfcc values for all audio files
    '''
    counter = 0
    df = pd.DataFrame(columns = ['mfcc_feature'])
    for index, path in enumerate(col):
        X, sample_rate = librosa.load(path, res_type = 'kaiser_fast',duration = 3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50)
        mfcc_scaled = np.mean(mfcc.T, axis = 0)
        df.loc[counter] = [mfcc_scaled]
        counter += 1
    
    mfcc_df = df.copy()
    mfcc_df.mfcc_feature = mfcc_df.mfcc_feature.apply(lambda x: separate(x))
    mfcc_list = ['mfcc'+str(i) for i in range(50)]
    
    df2 = pd.DataFrame(mfcc_df['mfcc_feature'].values.tolist(), columns = mfcc_list)
    return df2

