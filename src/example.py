import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from alperenet import AlpereNet

random_state = check_random_state(42)

# Load data from CSV
data = pd.read_csv('src/train.csv')

# Separate features and labels
X = data.drop(columns=['label'])  # Features
y = data['label']  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels
def one_hot_encode(labels, num_classes=10):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_classes, num_samples))
    one_hot_labels[labels, np.arange(num_samples)] = 1
    return one_hot_labels



X_train.shape, y_train.shape, X_test.shape, y_test.shape

X_train = X_train.T
X_test = X_test.T
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


AlpNet = AlpereNet()
AlpNet.add_layer(128, 784, activation=AlpereNet.leaky_relu, derivative=AlpereNet.derivative_leaky_relu)
AlpNet.add_layer(64, activation=AlpNet.leaky_relu, derivative=AlpNet.derivative_leaky_relu)
AlpNet.add_layer(32, activation=AlpNet.leaky_relu, derivative=AlpNet.derivative_leaky_relu)
AlpNet.add_layer(10, activation=AlpereNet.softmax, derivative=AlpereNet.softmax)


pred = AlpNet.forward(X_train)
pred = pd.DataFrame(pred)
pred = pred.T
pred.head()

losses2= AlpNet.fit(X_train, y_train, epochs=500, lr=0.5)


print(f'Test Accuracy: {np.mean(AlpNet.predict(X_test) == y_test.argmax(axis=0))}')
print(f'Train Accuracy: {np.mean(AlpNet.predict(X_train) == y_train.argmax(axis=0))}')
print(f'Final Loss: {losses2[-1]}')
print("Shapes of the data:")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

plt.plot(losses2)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

predictions = AlpNet.predict(X_test)
y_test_a = y_test.argmax(axis=0)
print(f'Accuracy: {np.mean(predictions == y_test_a)}')


preds = AlpNet.predict(X_train)
y_train_a = y_train.argmax(axis=0)
preds = pd.Series(preds)
y_train_a = pd.Series(y_train_a)
print("Predicted labels: ",preds.value_counts())
print("Real labels: ",y_train_a.value_counts())

import copy
weights_trained = copy.deepcopy(AlpNet.weights)
biases_trained = copy.deepcopy(AlpNet.biases)


def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

def plot_image_predictions(images, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Predicted: {predictions[i]}')
    plt.show()

plot_image_predictions(X_test.T, predictions)