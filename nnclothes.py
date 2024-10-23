import numpy as np
import matplotlib as plt
from tensorflow import keras
from tensorflow import layers

fashion_mnist = keras.dataset.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Using min-max scaling to normalize the data
X_train = X_train/255.0
X_test = X_test/255.0


# We create the NN with the input layer having 28x28 neurons
# There are 2 hidden layers with 128 and 64 neurons respectivly 

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()