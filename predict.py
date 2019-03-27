from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import random
import cv2
import numpy as np
import tensorflow
import skimage.io as io
from skimage.transform import resize
import tensorflow as tf

# TODO: find a good way to resize an image
# load a model
model = load_model('processes/134326032019/fruit360.h5py')
# fetch an image
img = io.imread('img/braeburn2.png', as_gray=False)
# prepare the image
img = img.astype('float32')
img = img / 255
img = img.reshape(-1, 100, 100, 3)
# predict and print
pred = model.predict_classes(img, verbose=1)
print(pred)
