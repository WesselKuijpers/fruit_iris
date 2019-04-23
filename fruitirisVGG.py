import time

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from abstract_network import Network

network = Network(epochs=20, batch_size=4, train_dir='data/plant_disease/train', val_dir='data/plant_disease/test', width=224, height=224)

# model
model = VGG19(pooling='max', weights=None, classes=38)       

model = network.set_model(model)
train_datagen = network.train_data_generator()
test_datagen = network.test_data_generator()
train_generator = network.train_directory_flow(train_datagen)
validation_generator = network.train_directory_flow(test_datagen)
hist = network.train(model, train_generator, validation_generator)
network.plot_graph(hist)