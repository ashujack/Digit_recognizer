import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)  # tensorflow channels_last
num_classes = 10


# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Train model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.fit(x_train, y_train, batch_size=512, epochs=30, verbose=2, callbacks=[learning_rate_reduction])

model.evaluate(x_test, y_test)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")