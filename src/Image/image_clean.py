import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import dataframe

df = pd.read_csv('../../data/fer2013.csv')

def clean_image_df():
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

