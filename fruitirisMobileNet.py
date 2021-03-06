import time

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.mobilenet_v2 import MobileNetV2

# hyperparameters
epochs = 5
batch_size = 1
# dataset directories
train_data_dir = 'data/Fruit/train'
validation_data_dir = 'data/Fruit/test'

# model
model = MobileNetV2(pooling='max', weights=None, classes=6)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train datagenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20)

# test datagenerator
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20)

# train datastreamer
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# validation datastreamer
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# try to train and save the model
try:
    hist = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=10 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1 // batch_size)

    model.save('saved_models/' + str(int(time.time())) + 'finished.h5py')
except KeyboardInterrupt:
    hist = None
    # if the process is interupted by the user save the interupted model
    model.save('saved_models/' + str(int(time.time())) + 'interupted.h5py')
    print("\ninterupted model was saved")
except:
    hist = None
    # re-raise the error on any other interuption
    raise
finally:
    # plot the process in a graph
    if hist:
        accuracy = hist.history['acc']
        val_accuracy = hist.history['val_acc']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'bo', label='Validation accuracy')
        plt.plot(epochs, accuracy, 'r')
        plt.plot(epochs, val_accuracy, 'b')

        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()
        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.plot(epochs, loss, 'r')
        plt.plot(epochs, val_loss, 'b')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
    else:
        print("No graph could be generated: DATA INCOMPLETE")
