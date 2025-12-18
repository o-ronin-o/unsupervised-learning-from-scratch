from typing import List, Dict

from layers import Layer
from learning_rate_scheduler import LearningRateScheduler
from loss import MSELoss
import numpy as np

class NeuralNetwork:
    """
    Neural network container that orchestrates layers.
    
    Parameters:
        layers: List of layer configurations
        learning_rate: Initial learning rate
        l2_lambda: L2 regularization strength
        scheduler: Learning rate scheduler
    """
    
    def __init__(self, layer_config, 
                 learning_rate: float = 0.01,
                 l2_lambda: float = 0.0,
                 scheduler = None):
        self.layers = self.build_neural_network(layer_config)
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.scheduler = scheduler
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
 
        
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    def update_parameters(self):
        if self.scheduler:
            lr = self.scheduler.get_lr()
        else:
            lr = self.learning_rate
        for layer in self.layers:
            layer.update_parameters(lr) 
    def compute_loss(self, y_true, y_pred):
        return MSELoss.forward(y_true, y_pred)
    


    def train_step(self, X_batch, y_batch):
       
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Compute loss
        loss = self.compute_loss(y_batch, y_pred)
        
        # Compute gradient of loss w.r.t predictions
        grad_loss = MSELoss.backward(y_batch, y_pred)
        
        # Backward pass
        self.backward(grad_loss)
        
        # Update parameters
        self.update_parameters()
        
        return loss

    def fit(self, X_train, y_train, epochs = 100, batch_size = 32 , verbose = True):
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            if self.scheduler:
                self.scheduler.step(epoch)
                lr = self.scheduler.get_lr()
            else:
                lr = self.learning_rate

            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            for i in range(0,n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                batch_loss = self.train_step(X_batch,y_batch)

                epoch_loss += batch_loss * X_batch.shape[0]

            epoch_loss /= n_samples
            self.history['train_loss'].append(epoch_loss)

            
            self.history['learning_rate'].append(lr)

            
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.6f}"
                print(msg)
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)


    def get_parameters(self) -> List[Dict]:
        """Get parameters of all layers."""
        return [layer.get_parameters() for layer in self.layers]


    def set_parameters(self, params: List[Dict]):
        """Set parameters of all layers."""
        for layer, param in zip(self.layers, params):
            layer.set_parameters(param)

    def build_neural_network(self, layer_config):
        """
        expects configration for the whole nn  for examples as follows: 
        layer_config = [
            {'n_inputs': 10, 'n_neurons': 16, 'activation': 'relu'},
            {'n_inputs': 16, 'n_neurons': 8, 'activation': 'relu'},
            {'n_inputs': 8, 'n_neurons': 1, 'activation': 'tanh'}
        ]
        """
        layers = []
        for config in layer_config:
            layer = Layer(
                n_inputs=config['n_inputs'],
                n_neurons = config['n_neurons'],
                activation=config.get('activation', 'relu'),
                l2_lambda=self.l2_lambda
            )
            layers.append(layer)
        return layers
        