from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import random
import numpy as np
import tensorflow
import skimage.io as io
from skimage.transform import resize
from keras.models import Sequential
import tensorflow as tf
from PIL import Image
from resizeimage import resizeimage

# TODO: find a good way to resize an image
# load a model
model = load_model('processes/DenseNet/densenet.h5py')
seq = Sequential()
seq.add(model)
seq.compile(loss='categorical_crossentropy', optimizer='adam')

location = input("INPUT: image location\n>>> ")

with open(location, 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [224, 224])
        cover.save(location, image.format)

# fetch an image
img = io.imread(location, as_gray=False)
# prepare the image
img = img.astype('float32')
img = img / 255
img = img.reshape(-1, 224, 224, 3)

# predict and print
pred = seq.predict_classes(img, verbose=1)
print(pred)
