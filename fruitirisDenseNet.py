from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from abstract_network import Network
from keras.applications.densenet import DenseNet121

network = Network(epochs=5, batch_size=32, train_dir='data/Fruit/train', val_dir='data/Fruit/test', width=224, height=224)

# model
model = DenseNet121(pooling='max', weights=None, classes=6)

model = network.set_model(model)
train_datagen = network.train_data_generator()
test_datagen = network.test_data_generator()
train_generator = network.train_directory_flow(train_datagen)
validation_generator = network.train_directory_flow(test_datagen)
hist = network.train(model, train_generator, validation_generator)
network.plot_graph(hist)
