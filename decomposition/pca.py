import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvalues = None
        self.components  = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance = None 

    def fit(self, X):
        """
        Fit PCA model to data using eigenvalue decomposition.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """

        # first we center the data 

        self.mean = np.mean(X,axis=0)
        X_centered = X - self.mean


        # then we compute the Covariance matrix
        # we need to reshape the samples to compute the cov using np 

        cov = np.cov(X_centered.T)

        # now that we have covariance matrix we calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # transposing eigen vectors (good practice)
        eigenvectors = eigenvectors.T

        #now we simply sort te eigen values to determine our components 
        idx = np.argsort(eigenvalues)[::-1] #reversing sorting order
        self.eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]

        # selecting the n_components 
        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])

        
        self.n_components  = min(self.n_components,eigenvectors.shape[0])
        self.components = eigenvectors[:self.n_components]

        # now we want to provide some transperancy regarding how much variance each component provide 
        total_variance  = np.sum(eigenvalues)
        self.explained_variance = self.eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / total_variance

        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
        return self
    
    def transform(self, X):
        """
        this method projects data features on principal components 
        """
        
        X_centered = X - self.mean
        return X_centered @ self.components 
    
    def inverse_transform(self, Z):
        """
        reconstructing by decoding the samples with the transpose of the principle components
        
        """
        
        return Z @ self.components.T + self.mean
    
    def reconstruction_error(self, X):
        
        """
        the basic logic is that we tranform the data using the fitted model 
        then decode it again and claculate the Mean Squared Error
        """
        Z = self.transform(X)

        X_reconstructed = self.inverse_transform(Z)

        mse = np.mean((X - X_reconstructed) ** 2)
        return mse
    
    # Helper results visualization functions 
    def get_explained_variance_summary(self):
        """
        Get a summary of explained variance.
        
        Returns:
            Dictionary with variance statistics
        """
        return {
            'eigenvalues': self.eigenvalues[:self.n_components],
            'explained_variance': self.explained_variance,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_explained_variance': self.cumulative_explained_variance,
            'total_variance': np.sum(self.eigenvalues)
        }
    
    def print_variance_summary(self):
        """
        Print formatted variance summary.
        """
        print("PCA Variance Summary")
        print("=" * 50)
        print(f"Total Components: {len(self.eigenvalues)}")
        print(f"Selected Components: {self.n_components}")
        print(f"Total Variance: {np.sum(self.eigenvalues):.4f}")
        print()
        print("Component-wise Variance:")
        print("-" * 50)
        
        for i in range(self.n_components):
            ratio_pct = self.explained_variance_ratio[i] * 100
            cum_pct = self.cumulative_explained_variance[i] * 100
            print(f"PC {i+1}: "
                  f"λ = {self.eigenvalues[i]:.4f}, "
                  f"Var% = {ratio_pct:.2f}%, "
                  f"Cum% = {cum_pct:.2f}%")
        
        print("=" * 50)

