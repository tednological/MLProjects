# sklearn will be the main library we use for its powerful ML capabilities. 
# StandardScaler is used to normalize the data
from sklearn.preprocessing import StandardScaler
# model_selection allows us to split up the data into training and testing set
from sklearn.model_selection import train_test_split
# This import allows us to easily use the KNN features built into sklearn
from sklearn.neighbors import KNeighborsClassifier
# This line imports the iris dataset!
from sklearn.datasets import load_iris
# This import allows us to calculate the accuracy of our KNN
from sklearn.metrics import accuracy_score
# This import allows us to easily find the confusion matrix for the KNN 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Pandas serves a valuable role, allowing us to use the Dataframe object
import pandas as pd


# importing in the dataset from sklearn
iris = load_iris()

# X will be the features
X = iris.data  
# Y will be the labels/classes
y = iris.target 

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split up the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# K = 1 is a good value to use for this KNN
knn = KNeighborsClassifier(n_neighbors=1)
# We use .fit to use X_train and y_train
knn.fit(X_train, y_train)

# We create a prediction set using the X_test variable
y_pred = knn.predict(X_test)

# Finally, we calculate the accuracy, comparing the test values against our y_pred
accuracy = accuracy_score(y_test, y_pred)
# Print out our accuracy
print(f"Model Accuracy: {accuracy * 100}%")

# Produces the confustion matrix for the KNN
conf_matrix = confusion_matrix(y_test, y_pred)

# Print that baby out!
print("Confusion Matrix:")
print(conf_matrix)
