import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from image_clean import *

training_df, validation_df, test_df, df = clean_image_df()


def fer2013_to_X(df):
    """Transforms the (blank separated) pixel strings in the DataFrame to an 3-dimensional array 
    (1st dim: instances, 2nd and 3rd dims represent 2D image)."""
    
    X = []
    pixels_list = df["pixels"].values
    
    for pixels in pixels_list:
        single_image = np.reshape(pixels.split(" "), (48, 48)).astype("float")
        X.append(single_image)
        
    # Convert list to 4D array:
    X = np.expand_dims(np.array(X), -1)
    
    # Normalize image data:
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    
    return X

# Training Data
X = fer2013_to_X(training_df)
y = pd.get_dummies(training_df['emotion']).values

# Save data
np.save("fer2013_X_train", X)
np.save("fer2013_y_train", y)

#Validation data
X_val = fer2013_to_X(validation_df)
y_val = pd.get_dummies(validation_df['emotion']).values
np.save("../../data/fer2013_X_val", X_val)
np.save("../../data/fer2013_y_val", y_val)

#Testing Data
X_test = fer2013_to_X(test_df)
y_test = pd.get_dummies(test_df['emotion']).values
np.save("../../data/fer2013_X_test", X_test)
np.save("../../data/fer2013_y_test", y_test)

# CNN
model = Sequential()
input_shape = (48,48,1)
model.add(Conv2D(64, (3, 3), input_shape=input_shape,activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Activation('softmax'))

#Compiling the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=Adam(lr=0.001))

earlystop = EarlyStopping(monitor='val_loss',
                          patience=20,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=20,
                              verbose=1,
                              min_delta=0.0001)

# Training Model
model.fit(x=X,     
            y=y, 
            batch_size=64, 
            epochs=100, 
            verbose=1, 
            validation_data=(X_val,y_val),
            shuffle=True,
            callbacks=[earlystop, reduce_lr]
            )

# Saving Model
model.save('fer2013.h5')
model.save_weights('fer2013_weights.h5')