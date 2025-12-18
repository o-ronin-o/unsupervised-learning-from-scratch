import numpy as np 
from activations import Activation

class Layer:
    def __init__(self, n_inputs, n_neurons , activation: str = 'relu', l2_lambda: float = 0.01):
        self.W = .10 * np.random.randn(n_inputs, n_neurons) #avoiding vanishing weights by inizialing weights (without systamtic approach) and avoiding exploding weights
        self.b = np.zeros((1, n_neurons))
        self.activation_name = activation
        self.l2_lambda = l2_lambda

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        #for backporpation we need a cache

        self.cache = {}

       
        self.activation_funcs = {
            'relu': (Activation.relu, Activation.relu_derivative),
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'tanh': (Activation.tanh, Activation.tanh_derivative),
            'linear': (Activation.linear, Activation.linear_derivative)
        }
        self.activation, self.activation_derivative = self.activation_funcs[activation]


    def forward(self, X):
        Z = np.dot(X, self.W) + self.b
        A = self.activation(Z)
        self.cache = {
            'X' : X, # the input
            'Z' : Z, # before activation
            'A' : A, # after activation
        }

        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
       
        batch_size = dA.shape[0]
        
        # Retrieving cache
        X = self.cache['X']  # (batch_size, input_dim)
        Z = self.cache['Z']  # (batch_size, output_dim)
        
        # Gradient w.r.t pre-activation
        dZ = dA * self.activation_derivative(Z)  # Element-wise, shape: (batch_size, output_dim)
        
        # Gradient w.r.t weights (average over batch)
        self.dW = (X.T @ dZ) / batch_size  # Shape: (input_dim, output_dim)
        
        # Gradient w.r.t biases (average over batch)
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size  # Shape: (1, output_dim)
        
        # Add L2 regularization gradient for weights only (not biases)
        if self.l2_lambda > 0:
            self.dW += self.l2_lambda * self.W
        
        # Gradient w.r.t input (to pass to previous layer)
        dX = dZ @ self.W.T  # Shape: (batch_size, input_dim)
        
        return dX
    
    def update_parameters(self, learning_rate: float):
        """Update weights and biases using gradient descent."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def get_parameters(self):
        """Return current parameters."""
        return {'W': self.W, 'b': self.b}
    
    def set_parameters(self, params):
        """Set parameters (for loading saved models)."""
        self.W = params['W']
        self.b = params['b']