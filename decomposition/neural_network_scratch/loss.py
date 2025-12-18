import numpy as np
class MSELoss:
    """Mean Squared Error loss."""
    
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute MSE loss.
        
        Args:
            y_true: True values, shape (batch_size, features)
            y_pred: Predicted values, shape (batch_size, features)
            
        Returns:
            MSE loss (scalar)
        """
        batch_size = y_true.shape[0]
        loss = np.sum((y_true - y_pred) ** 2) / batch_size
        return loss
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE loss w.r.t predictions.
        
        Args:
            y_true: True values, shape (batch_size, features)
            y_pred: Predicted values, shape (batch_size, features)
            
        Returns:
            Gradient w.r.t predictions, shape (batch_size, features)
        """
        batch_size = y_true.shape[0]
        grad = 2 * (y_pred - y_true) / batch_size
        return grad
