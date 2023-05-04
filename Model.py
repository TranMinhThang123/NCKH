from __future__ import division, print_function
import numpy as np
import os
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Conv1D, AvgPool2D, ReLU, Dense, add, Flatten, MaxPool2D, Dropout, MaxPooling1D, Activation, BatchNormalization, Lambda

model = keras.models.load_model("ECG_classification.h5")
# print(tf.__version__)
# img = cv2.imread("destination.png")
# print(type(img))
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img.shape)
# img = tf.image.resize(
#       img,
#       [224, 224]
#    )
# img = np.expand_dims(img, axis=0)
# print(img.shape)
# pred = model.predict(img)
# print(pred)