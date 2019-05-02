import keras
from keras.datasets import mnist
from keras.models import model_from_json
import matplotlib.pyplot as plt
import cv2, numpy as np
from random import randint

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)  # tensorflow channels_last
num_classes = 10

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
a = randint(0,10000)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')/255

# Predict
plt.imshow(x_test[a].reshape(28, 28))
pred = loaded_model.predict(x_test[a].reshape(1, img_rows, img_cols, 1))
print()
print("Loaded digit -> {}".format(pred.argmax()))
plt.show()