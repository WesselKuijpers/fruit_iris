from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import random
import cv2
import numpy as np
import tensorflow
import skimage.io as io
from skimage.transform import resize
from keras.models import Sequential
import tensorflow as tf

# TODO: find a good way to resize an image
# load a model
model = load_model('processes/DenseNet/densenet.h5py')
seq = Sequential()
seq.add(model)
seq.compile(loss='categorical_crossentropy', optimizer='adam')
# fetch an image
img = io.imread('img/pl5.png', as_gray=False)
# prepare the image
img = img.astype('float32')
img = img / 255
img = img.reshape(-1, 224, 224, 3)
# predict and print
pred = seq.predict_classes(img, verbose=1)
print(pred)
