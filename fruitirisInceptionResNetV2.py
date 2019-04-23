from abstract_network import Network
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2

network = Network(epochs=40, batch_size=16, train_dir='data/plant_disease/train',
                  val_dir='data/plant_disease/test', width=299, height=299)

# model
model = InceptionResNetV2(include_top=True, weights=None, classes=38)

model = network.set_model(model)
train_datagen = network.train_data_generator()
test_datagen = network.test_data_generator()
train_generator = network.train_directory_flow(train_datagen)
validation_generator = network.train_directory_flow(test_datagen)
hist = network.train(model, train_generator, validation_generator)
network.plot_graph(hist)
