from neural_network_scratch.neural_network import NeuralNetwork
from neural_network_scratch.learning_rate_scheduler import ExponentialLR
from neural_network_scratch.loss import MSELoss
import matplotlib.pyplot as plt
import numpy as np


class AutoEncoder:
    def __init__(self, input_dim, encoding_dim, encoder_config=None, learning_rate=0.1, l2_lambda=0.0001):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder_nn = None
        self.decoder_nn = None
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.learning_rate_gamma = 0.9995 
        self.loss = MSELoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [] 
        }

        if encoder_config is None:
            encoder_config = [
            {'n_inputs': self.input_dim, 'n_neurons': 32, 'activation': 'tanh'},
            {'n_inputs': 32, 'n_neurons': 16, 'activation': 'tanh'},
            {'n_inputs': 16, 'n_neurons': self.encoding_dim, 'activation': 'linear'}
            ]
        
        decoder_config = self.build_decoder_config(encoder_config)

        self.encoder_nn = NeuralNetwork(encoder_config, learning_rate=self.learning_rate, l2_lambda=self.l2_lambda, scheduler=ExponentialLR(self.learning_rate, self.learning_rate_gamma))
        self.decoder_nn = NeuralNetwork(decoder_config, learning_rate=self.learning_rate, l2_lambda=self.l2_lambda, scheduler=ExponentialLR(self.learning_rate, self.learning_rate_gamma))

    def encode(self, X):
        return self.encoder_nn.predict(X)
    
    def decode(self, Z):
        return self.decoder_nn.predict(Z)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = self.encode(X)
        X_recon = self.decode(Z)
        return X_recon

    def reconstruction_error(self, X, X_recon):
        return self.loss.forward(X, X_recon)
    
    def train_step(self, X_batch):
        y_batch  = X_batch
        
        # Forward
        Z = self.encoder_nn.forward(X_batch)
        X_recon = self.decoder_nn.forward(Z)

        # Loss
        loss = self.reconstruction_error(y_batch, X_recon)

        # Backward
        grad_X_recon = self.loss.backward(X_batch, X_recon)
        grad_Z = self.decoder_nn.backward(grad_X_recon)
        _ = self.encoder_nn.backward(grad_Z)
        
        # Update
        self.encoder_nn.update_parameters()
        self.decoder_nn.update_parameters()
        
        return loss

    def fit(self, X_train, epochs=800, batch_size=32, X_val=None, shuffle=True, verbose=True):
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_train = X_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                loss = self.train_step(X_batch)
                epoch_loss += loss
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            self.history['train_loss'].append(epoch_loss)

            if self.encoder_nn.scheduler:
                current_lr = self.encoder_nn.scheduler.get_lr()
            else:
                current_lr = self.learning_rate
            self.history['learning_rate'].append(current_lr)

            if X_val is not None:
                X_val_recon = self.reconstruct(X_val)
                val_loss = self.loss.forward(X_val, X_val_recon)
                self.history['val_loss'].append(val_loss)

            if self.encoder_nn.scheduler is not None:
                self.encoder_nn.scheduler.step(epoch)
                self.decoder_nn.scheduler.step(epoch)

            if verbose and epoch % 100 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.6f}")

    def set_gamma(self, x):
        self.learning_rate_gamma = x

    def build_decoder_config(self,encoder_config):
        decoder_config = []
        bottleneck_layer = encoder_config[-1]
        latent_dim = bottleneck_layer['n_neurons']
        reversed_encoder = list(reversed(encoder_config[:-1]))
        prev_dim = latent_dim

        for layer in reversed_encoder:

            decoder_config.append({
                'n_inputs': prev_dim,
                'n_neurons': layer['n_inputs'],
                'activation': layer['activation'] 
            })
            prev_dim = layer['n_inputs']

        decoder_config.append({
            'n_inputs': prev_dim,
            'n_neurons': encoder_config[0]['n_inputs'],
            'activation': 'linear'
        })
        return decoder_config

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if len(self.history['val_loss']) > 0:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Autoencoder Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history['learning_rate'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()