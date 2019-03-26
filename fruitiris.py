import time

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# hyperparametrs
epochs = 20
batch_size = 32
# dataset directories
train_data_dir = 'data/dataset/fruits-360/Training'
validation_data_dir = 'data/dataset/fruits-360/Test'

# model
model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(100, 100, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3)))
model.add(Dropout(0.1))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(11, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

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
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='categorical')

# validation datastreamer
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(100, 100),
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
    # if the process is interupted by the user save the interupted model
    if model.save('saved_models/' + str(int(time.time())) + 'interupted.h5py'):
        print("interupted model was saved")
    raise
except:
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

        z_acc = np.polyfit(epochs, accuracy, 1)
        p_acc = np.poly1d(z_acc)

        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, p_acc(epochs), 'g', label='accuracy trend')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        z_loss = np.polyfit(epochs, accuracy, 1)
        p_loss = np.poly1d(z_loss)
        
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, p_loss(epochs), 'g', label='loss trend')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
    else:
        print("No graph could be generated: DATA INCOMPLETE")
