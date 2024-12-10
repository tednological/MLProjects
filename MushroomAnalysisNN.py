# pandas allows us to read the csv
import pandas as pd
# numpy is good for data mangagement
import numpy as np
# matplotlib and seaborn are used for graphing data
import matplotlib.pyplot as plt

# sklearn is used for splitting up and normalizing data as well as some other minor functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# PyTorch is our main library. We use it for its powerful NN capabilites
import torch
import torch.nn as nn
# for optimization
import torch.optim as optim
# some misc tools for helping us process and move the data
from torch.utils.data import Dataset, DataLoader

# Since the mushroom data set has seperate files just for data and just for feature names I have placed all of the names here to avoid any complications
feature_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]
data = pd.read_csv('agaricus-lepiota.data', header=None, names=feature_names)

# One of the issues with this data set are the missing values. We can fix this by imputing the missing values with the mode
# The main perpetrator of missing values is the 'stalk-root' feature, which is missing about 3% of its data. Since relatively few are missing, 
# imputing the missing values with the mode is a good plan to ensure the data doesn't freak out on us 

# Replace '?' with NaN
data.replace('?', pd.NA, inplace=True)

# Calculate the mode of the 'stalk-root' feature
mode_stalk_root = data['stalk-root'].mode()[0]
print(f"Mode of 'stalk-root': {mode_stalk_root}")

# Impute the missing values with mode of 'stalk-root'
data['stalk-root'].fillna(mode_stalk_root, inplace=True)

# Now we check to make sure we aren't missing any values
missing_values = data.isnull().sum()
print("Missing Values per Feature:\n", missing_values)

# We are going to make a label encoder for every column in the data.columns
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# We seperate out the target variable from the features, setting the target to x and the features to y
X = data.drop('class', axis=1).values
y = data['class'].values  # 0 = edible, 1 = inedible


# Split into training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y
)

# We will go a bit further by also including a validation dataset on top of training and testing
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=50, stratify=y_train
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# We have to create a "Dataset" object for it to be loaded into a "DataLoader" for batching
class MushroomDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)  

    #returns the torch.tensor y
    def __len__(self):
        return len(self.y)
    # returns a specific item
    def __getitem__(self, a):
        return self.X[a], self.y[a]


# create the Dataset objects and fill them with the data
train_dataset = MushroomDataset(X_train, y_train)
val_dataset = MushroomDataset(X_val, y_val)
test_dataset = MushroomDataset(X_test, y_test)

# I set the batch size to 40 to make large batches to be processed
batch_size = 40

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# We will now define the nn using the MushroomClassifier class. nn.Module contains all of PyTorch's built-in functionalities
class MushroomClassifier(nn.Module):
    def __init__(self, input_size):
        # this line initializes the parent class (nn.Module)
        super(MushroomClassifier, self).__init__()
        # We use nn.Sequential as a container that sequences the layers in the order we need
        self.network = nn.Sequential(
            # transforms the input data into a new space with 64 dimensions
            nn.Linear(input_size, 64),
            # we then apply ReLU to help deal with the vanishing gradient problem
            nn.ReLU(),
            # acts as a regularizer to prevent overfitting
            nn.Dropout(0.5),
            # this hidden layer drops the dimensionality to capture higher-level patterns
            nn.Linear(64, 32),
            # once again for good effect
            nn.ReLU(),
            # regulizer
            nn.Dropout(0.5),
            # the final layer drops from 32 down to 1 for the binary classification. With one neuron it can either be on or off, or in our case, edible or inedible
            nn.Linear(32, 1)  
        )
    # x is the tensor that contains the feature data
    def forward(self, x):
        return self.network(x)  # outputs the logit either edible or inedible

# We take the shape to find input_size
input_size = X_train.shape[1]

# Initialize the model by calling MushroomClassifier
model = MushroomClassifier(input_size)

# we define the loss function and the optimizer. BCEWithLogitsLoss combines sigmoid with BCELoss 
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Because of how computationally expensive this nn will be, we want to make sure we are using the GPU, not CPU unless we really have to
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model.to(device)

# This number will be tuned to improve model accuracy
num_epochs = 60

# containers for loss and accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# We iterate num_epochs number of times, going back and forth to train the model
for epoch in range(num_epochs):
    # set the model to training mode
    model.train()
    # start the count of loss, correct prediction, and total predictions
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    # using the training DataLoader we iterate over batches of data
    for features, labels in train_loader:
        # this line moves the input features to the device, and also moves our labels to the device and reshapes them using .unsqueeze(1)
        # to add an extra dimension to match the output shape of the model
        features, labels = features.to(device), labels.to(device).unsqueeze(1)

        # clears the gradients of all torch.Tensor objects, needed for the backpropagation
        optimizer.zero_grad()

        # this is the forward pass, starting with the outputs from the model
        outputs = model(features)
        # We then apply the loss function to compute the loss using the outputs and labels
        loss = criterion(outputs, labels)

        # run back through it, optimizing at each step using the computer gradients
        loss.backward()
        optimizer.step()

        # multiply the loss by the total number of samples to find total loss, then add it ot the running loss
        running_loss += loss.item() * features.size(0)

        # This block calculates the accuracy of the model
        # we apply the sigmoid activation function to the outputs, giving us probabilities between 0 and 1
        preds = torch.sigmoid(outputs)
        # we establish a threshold (.5) to convert the continous probablities to discrete classes
        predicted = (preds > 0.5).float()
        # correct predictions!
        correct_preds += (predicted == labels).sum().item()
        # add the number of samples in this batch to the running total
        total_preds += labels.size(0)

    # we then calculate the average loss and the accuracy per sample for the entire epoch
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_preds / total_preds
    #we store the average training loss per epoch in train_losses and the epoch accuracy in train_accuracies
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # We will now validate the model using the data we set aside earlier
    # set the model to evaluation mode
    model.eval()
    # initialize validation variables
    val_running_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0
    # we disable gradient calculation to reduce computational space (my laptop cannot handle the model without this XD)
    with torch.no_grad():
        # go through again like before, except we now have the val_loader instead of train_loader
        for features, labels in val_loader:
            # move the data to the device and reshape labels like before
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            # FORWARD PASS, we pass the validation features in to obtain the predictions
            outputs = model(features)
            # find the loss between predictions and the true labels
            loss = criterion(outputs, labels)
            #find total loss
            val_running_loss += loss.item() * features.size(0)
            # once again we turn them into probabilites, then classes, then find out how many were correct
            preds = torch.sigmoid(outputs)
            predicted = (preds > 0.5).float()
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)
    # once again we then calculate the average loss and the accuracy per sample for the entire epoch
    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_correct_preds / val_total_preds
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)
    # outputs the performance metrics of the current epoch
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | '
          f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')

# The plots! 
# We create a chart for the loss
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Run Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Run Loss')
plt.xlabel('Epochs Passed')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
# this is the chart for the accuracy
plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch Passed')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

# show the plots!
plt.show()

# we will now evaluate the model, seeing how it performs on unseen data
# we take in the module, the dataloader, and the device we are going to process this on
def evaluate_model(model, dataloader, device):
    # set to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    # once again disable gradient computation to save processing power and memory
    with torch.no_grad():
        # loop through each batch using the dataloader
        for features, labels in dataloader:
            # put the features and labels on the device and unsqueeze our labels
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            # obtain the outputs 
            outputs = model(features)
            # create probs using sigmoid
            preds = torch.sigmoid(outputs)
            # convert probs to classes
            predicted = (preds > 0.5).float()
            # put everything where it needs to be using numpy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # return the numpy arrays 
    return np.array(all_preds), np.array(all_labels)

# we get the predictions and true labels from the test set
test_preds, test_labels = evaluate_model(model, test_loader, device)

# compute the various metrics and make the confusion matrix
test_accuracy = accuracy_score(test_labels, test_preds)
conf_matrix = confusion_matrix(test_labels, test_preds)
class_report = classification_report(test_labels, test_preds, target_names=['Edible', 'Poisonous'])
roc_auc = roc_auc_score(test_labels, test_preds)

# Display the results, including our confusion matrix and constructed classification report
print(f"Accuracy: {test_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix")
print(conf_matrix)
print("Class Report")
print(class_report)

# Next up, we need to plot the ROC curve. To do this, we need to reevaluate the model and 
# get the probablties and associated labels
def get_predictions(model, dataloader, device):
    # Set to evaluation mode
    model.eval()  
    all_probs = []
    all_labels = []
    # no_grad again for computational reasons
    with torch.no_grad():
        # use the dataloader again
        for features, labels in dataloader:
            # send features to device
            features = features.to(device)
            # send labels to the device
            labels = labels.to(device)
            # get the outputs
            outputs = model(features)
            # sigmoid into probablities
            probs = torch.sigmoid(outputs)
            # add probs to all_probs
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # return the two numpy arrays
    return np.array(all_labels), np.array(all_probs).flatten()

# Obtain true labels and predicted probabilities
y_true, y_probs = get_predictions(model, test_loader, device)

# Compute ROC curve metrics
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = roc_auc_score(y_true, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--', label='Random Guessing')

