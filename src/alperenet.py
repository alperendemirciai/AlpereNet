# # let's make neural network from scratch using numpy
# ### Author: Alperen Demirci
# ### Mail: alperendemirci65@gmail.com 
# ### Date: 23-03-2024

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

class AlpereNet:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x) + 1e-6)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def leaky_relu(x):
        return np.maximum(0.01*x, x)
    
    @staticmethod
    def derivative_leaky_relu(x):
        return 0.01 * (x < 0) + 1 * (x > 0)
    
    @staticmethod
    def softmax(x):
        exps = np.exp(x)
        return exps / (np.sum(exps, axis=0)+1e-6)

    @staticmethod
    def derivative_sigmoid(x):
        return x * (1 - x)
    
    @staticmethod
    def derivative_relu(x):
        return 1 * (x > 0)
    
    @staticmethod
    def derivative_tanh(x):
        return 1 - x**2

    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.activation = []
        self.derivative = []

    def add_layer(self, units, input_shape=None,activation=sigmoid, derivative=derivative_sigmoid):
        if input_shape:
            self.weights.append(np.random.randn(units, input_shape)*0.01)
        else:
            self.weights.append(np.random.randn(units, self.weights[-1].shape[0])*0.01)
        self.biases.append(np.random.randn(units,1)*0.01)
        self.layers.append(np.zeros((units, 1)))
        self.activation.append(activation)
        self.derivative.append(derivative)

    def random_init(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.randn(self.weights[i].shape[0], self.weights[i].shape[1])*0.01
            self.biases[i] = np.random.randn(self.biases[i].shape[0], 1)*0.01

    def forward(self, X):
        if(len(self.layers)==len(self.weights)):
            self.layers.insert(0, X)
        else:
            self.layers[0] = X
        for i in range(1,len(self.weights)):
            self.layers[i] = self.activation[i-1](np.dot(self.weights[i-1], self.layers[i-1]) + self.biases[i-1])
        self.layers[-1] = self.softmax(np.dot(self.weights[-1], self.layers[-2]) + self.biases[-1])
        return self.layers[-1]
    
    def backward(self, y):
        m = y.shape[1]
        deltas = []
        deltas.append((self.layers[-1] - y)*1/m)
        for i in range(len(self.layers)-2, 0, -1):
            deltas.insert(0, np.dot(self.weights[i].T, deltas[0]) * self.derivative[i-1](self.layers[i]))
        return deltas
    
    def update(self, deltas, lr):
        changes = []
        for i in range(len(self.weights)):
            changes.append(np.sum(np.dot(deltas[i], self.layers[i].T)))      
            self.weights[i] -= lr * np.dot(deltas[i], self.layers[i].T)
            self.biases[i] -= lr * np.sum(deltas[i], axis=1, keepdims=True)
        return self.weights, self.biases, changes

    def calculate_loss(self,Y, Y_hat):
        m = Y.shape[1]
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat+1e-6)))
        L = -(1/m) * L_sum
        return L
        
    def fit(self, X, y, epochs=100, lr=0.01):
        print("Training started!")
        print("Epochs: ",epochs)
        print("Learning Rate: ",lr)
        self.random_init()
        loss = []
        for epoch in range(epochs):
            self.forward(X)
            deltas = self.backward(y)
            for i in range(len(self.weights)):
                self.weights[i] -= lr * np.dot(deltas[i], self.layers[i].T)
                self.biases[i] -= lr * np.sum(deltas[i], axis=1, keepdims=True)
         
            if epoch % 10 == 0:
                print(f'Epoch {epoch} Loss: {self.calculate_loss(y, self.layers[-1])}')
                loss.append(self.calculate_loss(y, self.layers[-1]))
        return loss
    
    def predict(self, X):
        return self.forward(X).argmax(axis=0)
