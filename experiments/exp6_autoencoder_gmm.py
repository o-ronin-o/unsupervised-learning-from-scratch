import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'decomposition'))
sys.path.append(os.path.join(parent_dir, 'decomposition', 'neural_network_scratch'))

from decomposition.auto_encoder import AutoEncoder
from clustering.GMM import GMM
from utils.metrics import purity_score

def get_bic(model, X, cov_type, k):
    """
    Calculate BIC (Bayesian Information Criterion).
    Lower is better.
    """
    n, d = X.shape
    # Safety check: if model didn't converge or failed, return infinity
    if not model.log_likelihoods:
        return float('inf')
        
    ll = model.log_likelihoods[-1]
    
    # Count parameters based on covariance type
    if cov_type == 'full': 
        # Means + Covariances (matrix) + Weights
        params = k*d + k*(d*(d+1)//2) + (k-1)
    elif cov_type == 'diag': 
        # Means + Covariances (diagonal) + Weights
        params = k*d + k*d + (k-1)
    elif cov_type == 'tied': 
        # Means + Covariances (shared matrix) + Weights
        params = k*d + (d*(d+1)//2) + (k-1)
    else: # spherical
        # Means + Covariances (single number) + Weights
        params = k*d + k + (k-1)
        
    return params * np.log(n) - 2 * ll

# Load Data
print("Loading Data...")
data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)
y = data.target

# Setup
bottleneck_sizes = [2, 5, 10, 15, 20]
cov_types = ['full', 'tied', 'diag', 'spherical']
k = 2  # We know there are 2 classes (Benign vs Malignant)

print(f"\nStarting Experiment 6 (Autoencoder + GMM)...")
print("-" * 75)
print(f"{'Dim':<5} | {'Shape':<10} | {'BIC (Lower=Better)':<20} | {'Purity':<10}")
print("-" * 75)

best_configs = []

for dim in bottleneck_sizes:
    # Train Autoencoder
    # Using LR=0.1, Epochs=800
    ae = AutoEncoder(input_dim=X_scaled.shape[1], encoding_dim=dim, 
                     learning_rate=0.1)
    
    # We suppress the training prints (verbose=False) to keep the table clean
    # But trust that it IS training for 800 epochs in the background!
    ae.fit(X_scaled, epochs=800, batch_size=32, verbose=False)
    
    # Encode (Compress the data)
    X_encoded = ae.encode(X_scaled)
    
    local_best_bic = float('inf')
    local_best_shape = ''
    
    for cov in cov_types:
        try:
            # Run GMM on compressed data
            gmm = GMM(k=k, dim=dim, covariance_type=cov, max_iter=100)
            gmm.fit(X_encoded)
            
            # Score
            bic = get_bic(gmm, X_encoded, cov, k)
            
            # Calculate Purity
            # Note: GMM.z gives probabilities, we take argmax to get hard labels
            labels = np.argmax(gmm.z, axis=1)
            pur = purity_score(y, labels)
            
            print(f"{dim:<5} | {cov:<10} | {bic:.1f}{'':<15} | {pur:.4f}")
            
            if bic < local_best_bic:
                local_best_bic = bic
                local_best_shape = cov
                
        except Exception as e:
            # If GMM fails (singular matrix), just report failure
            print(f"{dim:<5} | {cov:<10} | FAILED")
            
    best_configs.append(local_best_shape)
    print("-" * 75)
    if local_best_shape:
        print(f">> WINNER for {dim} dims: {local_best_shape.upper()}")
    else:
        print(f">> WINNER for {dim} dims: NONE")
    print("-" * 75)

# Plotting the Winners
plt.figure(figsize=(10, 5))
# Draw grey bars as background
plt.bar([str(d) for d in bottleneck_sizes], [1]*5, color='#e0e0e0')

plt.title("Exp 6: Best GMM Shape per Autoencoder Dimension")
plt.xlabel("Bottleneck Dimension")
plt.ylabel("Winner")
plt.yticks([]) # Hide y-axis numbers since they don't mean anything here

# Write the names of the winners inside the bars
for i, (dim, shape) in enumerate(zip(bottleneck_sizes, best_configs)):
    if shape:
        text = shape.upper()
    else:
        text = "N/A"
        
    plt.text(i, 0.5, text, ha='center', va='center', 
             fontweight='bold', fontsize=12, color='black')

plt.show()