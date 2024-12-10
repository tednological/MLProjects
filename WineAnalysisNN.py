# Import libraries
# We use pandas for its powerful dataframe structure and read_csv function
import pandas as pd
# We use matplotlib to visualize the data
import matplotlib.pyplot as plt
# Numpy arrays are powerful tools for storing a performing operations on large groups of data
import numpy as np
# Used for splitting data
from sklearn.model_selection import train_test_split
# used for preprocessing the data
from sklearn.preprocessing import StandardScaler
# Used for finding the RMSE
from sklearn.metrics import mean_squared_error
# tensorflow is our main machine learning library. It offers methods for constructing and optimizing neural networks. 
import tensorflow as tf
from tensorflow.keras import layers, models

# Read the Red Wine
red_wine = pd.read_csv('winequality-red.csv', sep=';')
red_wine['wine_type'] = 0  # 0 for red

# Read the White Wine
white_wine = pd.read_csv('winequality-white.csv', sep=';')
white_wine['wine_type'] = 1  # 1 for white

# To assess the quality of both red and white wine, we put them in the same data set
wine_dataset = pd.concat([red_wine, white_wine], axis=0)

# create variables for target and other features
X = wine_dataset.drop('quality', axis=1)
y = wine_dataset['quality']

# Scale features the features using StandardScaler
scaler = StandardScaler()
XScaled = scaler.fit_transform(X)

# Split into train/test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(XScaled, y, test_size=0.2, random_state=50)

# initialize rmse_scores
rmsescores = []



# Define the model
model = models.Sequential()
# The first layer has an input equal to the number of features. It goes into a layer of 75 neurons
# we use ReLU as the activation 
model.add(layers.Dense(75, activation='relu', input_shape=(Xtrain.shape[1],)))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1))  

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
trainedModel = model.fit(Xtrain, Ytrain, validation_split=0.2, epochs=30, batch_size=30)

# Evaluate on test set
test_loss, test_mae = model.evaluate(Xtest, Ytest, verbose=0)
print("Test MAE:", test_mae)

# Predict quality
predictions = model.predict(Xtest)
print("Sample predictions:", predictions[:5])

# picking a few different epoch counts to see how things shape up
epochCounts = [20, 40, 80]

for e in epochCounts:
    # Define the model
    model = models.Sequential()
    # The first layer has an input equal to the number of features. It goes into a layer of 75 neurons
    # we use ReLU as the activation 
    model.add(layers.Dense(75, activation='relu', input_shape=(Xtrain.shape[1],)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))  
    # use adam as the optimizer and mse as loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # now we train this model using the chosen epoch count
    history = model.fit(Xtrain, Ytrain, validation_split=0.2, epochs=e, batch_size=30, verbose=0)
    
    # check how well this model does on our test data
    predictions = model.predict(Xtest)
    mse = mean_squared_error(Ytest, predictions)
    rmse = np.sqrt(mse)
    rmsescores.append(rmse)



# now let's make ourselves a plot to see how rmse changes with epochs
plt.figure(figsize=(8,5))
plt.plot(epochCounts, rmsescores, marker='o')
plt.title('rmse vs. epoch count')
plt.xlabel('number of epochs')
plt.ylabel('rmse')
plt.grid(True)
plt.show()
