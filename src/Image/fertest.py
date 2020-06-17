import sys, os
import pandas as pd
import numpy as np
from keras.models import load_model


#Load Image Recognition Convolutional Neural Network
model = load_model('../../data/fer2013.h5')
model.load_weights('../../data/fer2013_weights.h5')

#Set list for y and y_predicted
y = []
y_pred = []

# Load in X_test and y_test arrays using NumPy
X_test = np.load('../../data/fer2013_X_test.npy')
y_test = np.load('../../data/fer2013_y_test.npy')


# Initialize y_hat as the values predicted from X_test (our test predictors)
y_hat= model.predict(X_test)

# Convert it into a list
y_h = y_hat.tolist()
y_t = y_test.tolist()
count = 0

# Calculates the highest predictions for the imported image file and appends it to a list
for i in range(len(y_test)):
    yy = max(y_h[i])
    yyt = max(y_t[i])
    y_pred.append(y_h[i].index(yy))
    y.append(y_t[i].index(yyt))
    if(y_h[i].index(yy)== y_t[i].index(yyt)):
        count+=1

acc = (count/len(y_test))*100

#saving values for confusion matrix and analysis
np.save('../../data/y_true_test', y)
np.save('../../data/y_pred_test', y_pred)
print("Predicted and true label values saved")
print("Accuracy on test set :"+str(acc)+"%")