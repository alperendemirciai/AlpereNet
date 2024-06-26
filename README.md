# AlpereNet: Neural Network Implementation

AlpereNet is a simple neural network implementation built using NumPy, a fundamental package for scientific computing in Python. This project provides a straightforward neural network architecture with customizable activation functions.

## Features

- Implements a feedforward neural network with customizable activation functions including sigmoid, ReLU, and tanh.
- Provides methods for forward pass, backward pass, and model training.
- Allows customization of network architecture by adding layers with a specified number of units and activation functions.

## Requirements

- Python 3.x
- NumPy

## Installation

You can install NumPy using pip:

```bash
pip install numpy
```

## Usage

1. Import the `AlpereNet` class from the `alperenet.py` file into your Python script.
2. Create an instance of the `AlpereNet` class.
3. Add layers to the network using the `add_layer` method.
4. Train the model using the `fit` method with your training data.
5. Predict using the `predict` method with new input data.

Example usage:

```python
from alperenet import AlpereNet
import numpy as np

# Create an instance of AlpereNet
model = AlpereNet()

# Add layers to the network
model.add_layer(64, input_shape=128, activation=AlpereNet.relu)
model.add_layer(32, activation=AlpereNet.relu)
model.add_layer(10, activation=AlpereNet.softmax)

# Generate some random data
X_train = np.random.randn(128, 1000)
y_train = np.random.randint(0, 10, (10, 1000))

# Train the model
loss_history = model.fit(X_train, y_train, epochs=100, lr=0.01)

# Make predictions
X_test = np.random.randn(128, 100)
predictions = model.predict(X_test)
```
* You can check the `AlpereNet.ipynb` for the demonstration of usage with MNIST dataset.
* Also you can check `src/example.py` for the script usage on MNIST instead of Jupyter Notebook.

## Results 
* Achieved **%95.75** Accuracy on training data on **1000** epochs and learning rate as **0.5**.
* Architecture used was **(784x128,ReLU) -> (128x64,ReLU) -> (64x32,ReLU) -> (32x10,Softmax)**
* Training process completed about **4 minutes** on Apple M3 Pro 11C on CPU.

## Media
### Here are some of the predicted handwritten digits!
![Here are some of the predicted handwritten digits!](plots/predictions.png)

### Loss - Epoch plot 
![Loss - Epoch plot](plots/epoch_loss.png)

  
## Contributing

Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests through GitHub.
