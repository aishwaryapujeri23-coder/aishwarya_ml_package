"""All activation functions + their derivatives."""
import numpy as np


def relu(z):
    """ReLU: max(0, z). Used in hidden layers."""
    return np.maximum(0, z)

def relu_d(z):
    """Derivative of ReLU."""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid: 1/(1+e^-z). Output in (0,1). Good for binary classification."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_d(z):
    """Derivative of sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    """Tanh: output in (-1,1). Zero-centered, good for hidden layers."""
    return np.tanh(z)

def tanh_d(z):
    """Derivative of tanh."""
    return 1 - np.tanh(z) ** 2

def linear(z):
    """Linear: identity. Used for regression output."""
    return z

def linear_d(z):
    return np.ones_like(z)

ACTIVATIONS = {
    "relu"   : (relu,    relu_d),
    "sigmoid": (sigmoid, sigmoid_d),
    "tanh"   : (tanh,    tanh_d),
    "linear" : (linear,  linear_d),
}
