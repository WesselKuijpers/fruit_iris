import time

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19

# hyperparameters
epochs = 5
batch_size = 1
# dataset directories
train_data_dir = 'data/Fruit/train'
validation_data_dir = 'data/Fruit/test'

# model
# model = Sequential()
# model.add(Conv2D(128, (3, 3), input_shape=(100, 100, 3)))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(128, (3, 3)))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(128, (3, 3)))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(128, (3, 3)))
# model.add(Dropout(0.1))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.4))

# model.add(Flatten())
# model.add(Dense(128))
# model.add(LeakyReLU(alpha=0.1))
# model.add(Dense(11, activation='relu'))
# model.add(Dense(6, activation='softmax'))

# compile the model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='softmax'))

vgg = VGG19(pooling='max')
vgg.layers.pop()
model = Sequential()
model.add(vgg)
model.add(Dense(6, activation='softmax'))
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
        steps_per_epoch=5516 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1883 // batch_size)

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
