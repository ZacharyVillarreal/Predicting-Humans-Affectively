import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import dataframe
df = pd.read_csv('../../data/fer2013.csv')



def clean_image_df():
    '''
    clean_image_df: takes in raw fer2013 dataframe. Creates an emotion
    labeled column that corresponds to numerical representations. Then splits
    up the df into three dataframes: Testing, Training, and Validation.

    Parameters
    ----------
    df: pandas dataframe

    Returns
    -------
    Training_df, Validation_df, Test_df, df: training data, validation data, testing data, and original data
    ''' 
    df = pd.read_csv('../../data/fer2013.csv')
    Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    lst = []
    for i in df.emotion:
        lst.append(Emotions[i])

    df['emotion_label'] = lst
    training_df = df[df['Usage'] == 'Training']
    validation_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df.Usage == 'PrivateTest']
    
    return training_df, validation_df, test_df, df

