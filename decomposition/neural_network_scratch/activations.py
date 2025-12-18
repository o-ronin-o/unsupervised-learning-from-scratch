import numpy as np 
class Activation:
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)
    
    @staticmethod 
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x>0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x,-50,50)))
    

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(x)
        return s*(1-s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh: 1 - tanh²(x)"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation (no transformation)"""
        return x
    
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of linear activation: 1"""
        return np.ones_like(x)
