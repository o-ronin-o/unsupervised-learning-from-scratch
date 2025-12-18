from neural_network_scratch.neural_network import NeuralNetwork
from neural_network_scratch.learning_rate_scheduler import ExponentialLR
from neural_network_scratch.loss import MSELoss
import matplotlib.pyplot as plt
import numpy as np


class AutoEncoder:
    def __init__(self, input_dim, encoding_dim,encoder_config = None, learning_rate = .01 ,l2_lambda = 0.0001):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder_nn = None
        self.decoder_nn = None
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.learning_rate_gamma = .9
        self.loss = MSELoss()
        if encoder_config is None:
            encoder_config = [
            {'input_dim': self.input_dim, 'output_dim': 64, 'activation': 'relu'},
            {'input_dim': 64, 'output_dim': 32, 'activation': 'relu'},
            {'input_dim': 32, 'output_dim': 16, 'activation': 'relu'},
            {'input_dim': 16, 'output_dim': self.encoding_dim, 'activation': 'linear'}
            ]
        decoder_config = self.build_decoder_config(encoder_config)

        self.encoder_nn = NeuralNetwork(encoder_config,learning_rate = self.learning_rate, l2_lambda = self.l2_lambda, scheduler=ExponentialLR(self.learning_rate, self.learning_rate_gamma))
        self.decoder_nn = NeuralNetwork(decoder_config,learning_rate = self.learning_rate, l2_lambda = self.l2_lambda, scheduler=ExponentialLR(self.learning_rate, self.learning_rate_gamma))

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    
    
    def encode(self, X):
        """
        Encode input data to latent space.
        """
        return self.encoder_nn.predict(X)
    
    
    def decode(self, Z):
        """
        Decode latent representation back to original space.
        """
        return self.decoder_nn.predict(Z)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data: encode then decode.
        
        """
        Z = self.encode(X)
        X_recon = self.decode(Z)
        return X_recon

    
    def reconstruction_error(self, X, X_recon):
        return self.loss.forward(X, X_recon)

    
    
    def train_step(self, X_batch):
        """
        Single training step for autoencoder.
        """
        y_batch  = X_batch
        # Forward pass
        Z = self.encoder_nn.forward(X_batch)
        
        X_recon = self.decoder_nn.forward(Z)

        # Compute loss
        loss = self.reconstruction_error(y_batch, X_recon)

        grad_X_recon = self.loss.backward(X_batch, X_recon)

        grad_Z = self.decoder_nn.backward(grad_X_recon)
        _ = self.encoder_nn.backward(grad_Z)
        
        return loss
    

    def fit(
        self,
        X_train,
        epochs=100,
        batch_size=32,
        X_val=None,
        shuffle=True,
        verbose=True
    ):
        """
        Train the autoencoder using mini-batch gradient descent.
        """

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # ===== Shuffle =====
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_train = X_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            # ===== Mini-batch training =====
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]

                loss = self.train_step(X_batch)

                epoch_loss += loss
                n_batches += 1

            epoch_loss /= n_batches
            self.history['train_loss'].append(epoch_loss)

            # ===== Validation =====
            if X_val is not None:
                X_val_recon = self.reconstruct(X_val)
                val_loss = self.loss.forward(X_val, X_val_recon)
                self.history['val_loss'].append(val_loss)

            # ===== Learning rate scheduling =====
            if self.encoder_nn.scheduler is not None:
                self.encoder_nn.scheduler.step()
                self.decoder_nn.scheduler.step()

            # ===== Logging =====
            if verbose:
                if X_val is not None:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Train Loss: {epoch_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Train Loss: {epoch_loss:.6f}"
                    )


    def set_gamma(self, x):
        self.learning_rate_gamma = x

    def build_decoder_config(self,encoder_config):
        """
        Build decoder layer configuration dynamically from encoder configuration.

        Args:
            encoder_config (list of dict): Encoder layer configurations

        Returns:
            list of dict: Decoder layer configurations
        """
        decoder_config = []

        # The bottleneck layer is the last encoder layer
        bottleneck_layer = encoder_config[-1]
        latent_dim = bottleneck_layer['output_dim']

        # Reverse encoder layers EXCEPT the bottleneck
        reversed_encoder = list(reversed(encoder_config[:-1]))

        prev_dim = latent_dim

        # Build decoder hidden layers
        for layer in reversed_encoder:
            decoder_config.append({
                'input_dim': prev_dim,
                'output_dim': layer['input_dim'],
                'activation': layer['activation']  # reuse hidden activation
            })
            prev_dim = layer['input_dim']

        # Final reconstruction layer
        decoder_config.append({
            'input_dim': prev_dim,
            'output_dim': encoder_config[0]['input_dim'],
            'activation': 'linear'  # reconstruction layer
        })

        return decoder_config
    #helper funtion
    def plot_training_history(self):
        """Plot training and validation loss."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if len(self.history['val_loss']) > 0:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Autoencoder Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rates
        ax2.plot(self.encoder_nn.history['learning_rate'])
        ax2.plot(self.decoder_nn.history['learning_rate'])

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
