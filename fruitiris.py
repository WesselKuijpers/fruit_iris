import time

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from abstract_network import Network

network = Network(epochs=20, batch_size=32, train_dir='data/plant_disease/train', val_dir='data/plant_disease/test', width=256, height=256)

# model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same",input_shape=(256, 256, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(38))
model.add(Activation("softmax"))

model = network.set_model(model)
train_datagen = network.train_data_generator()
test_datagen = network.test_data_generator()
train_generator = network.train_directory_flow(train_datagen)
validation_generator = network.train_directory_flow(test_datagen)
hist = network.train(model, train_generator, validation_generator)
network.plot_graph(hist)
