import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Using min-max scaling to normalize the data
X_train = X_train/255.0 
X_test = X_test/255.0

# Flatten out the data into vectors of size 784
X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat = X_test.reshape(-1, 28 * 28)

# We create the NN with the input layer having 28x28 neurons
# There are 2 hidden layers with 128 and 64 neurons respectivly 

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Compile the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the flattened X and the y_training
history = model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Test the model using the .evaluate command
test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=2)
print('\nTest accuracy:', test_acc)


# Get predictions for the first 5 test images
predictions = model.predict(X_test_flat[:5])

# Predict classes
y_pred = np.argmax(model.predict(X_test_flat), axis=1)


# Print everything and show some of the predictions
for i in range(5):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Actual: {class_names[y_test[i]]} | Predicted: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
    plt.show()
    
    
