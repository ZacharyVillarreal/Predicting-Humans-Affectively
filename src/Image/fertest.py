import sys, os
import pandas as pd
import numpy as np
from keras.models import load_model

model = load_model('../../data/fer2013.h5')
model.load_weights('../../data/fer2013_weights.h5')

y = []
y_pred = []

X_test = np.load('../../data/fer2013_X_test.npy')
y_test = np.load('../../data/fer2013_y_test.npy')


y_hat= model.predict(X_test)
y_h = y_hat.tolist()
y_t = y_test.tolist()
count = 0

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