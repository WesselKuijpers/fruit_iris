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

model = load_model('110525032019/fruit360_v1.h5py')
img = io.imread('img/rd_1.png', as_gray=False)
# img = resize(img, (100, 100, 3))
img = img.astype('float32')
img = img / 255
img = img.reshape(-1, 100, 100, 3)
pred = model.predict_classes(img, verbose=1)
print(pred)
