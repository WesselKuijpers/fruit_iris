from abstract_network import Network
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

network = Network(epochs=5, batch_size=32, train_dir='data/Fruit/train', val_dir='data/Fruit/test')

# model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model = network.set_model(model)
train_datagen = network.train_data_generator()
test_datagen = network.test_data_generator()
train_generator = network.train_directory_flow(train_datagen)
validation_generator = network.train_directory_flow(test_datagen)
hist = network.train(model, train_generator, validation_generator)
network.plot_graph(hist)