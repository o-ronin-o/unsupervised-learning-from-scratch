import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from decomposition.pca import PCA
from clustering.GMM import GMM
from utils.metrics import purity_score

# Helper for BIC calculation
def get_bic(model, X, cov_type, k):
    n, d = X.shape
    ll = model.log_likelihoods[-1]
    
    # Count parameters
    if cov_type == 'full': params = k*d + k*(d*(d+1)//2) + (k-1)
    elif cov_type == 'diag': params = k*d + k*d + (k-1)
    elif cov_type == 'tied': params = k*d + (d*(d+1)//2) + (k-1)
    else: params = k*d + k + (k-1)
        
    return params * np.log(n) - 2 * ll

# Load Data
data = load_breast_cancer()
X = data.data
y = data.target
X_scaled = StandardScaler().fit_transform(X)

# Setup Tournament
components_list = [2, 5, 10, 15, 20]
cov_types = ['full', 'tied', 'diag', 'spherical']
k = 2  # there are 2 classes

results = []

print(f"Starting Tournament (GMM + PCA)...")
print("-" * 60)
print(f"{'Dim':<5} | {'Shape':<10} | {'BIC (Lower=Better)':<20} | {'Purity':<10}")
print("-" * 60)

best_shapes = []

for n in components_list:
    # Simplify Data
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    local_best_bic = float('inf')
    local_best_shape = ''
    
    for cov in cov_types:
        try:
            # Run GMM
            gmm = GMM(k=k, dim=n, covariance_type=cov, max_iter=100)
            gmm.fit(X_pca)
            
            # Score
            bic = get_bic(gmm, X_pca, cov, k)
            
            # Get Purity (Hard labels)
            labels = np.argmax(gmm.z, axis=1)
            pur = purity_score(y, labels)
            
            print(f"{n:<5} | {cov:<10} | {bic:.1f}{'':<15} | {pur:.4f}")
            
            if bic < local_best_bic:
                local_best_bic = bic
                local_best_shape = cov
                
        except Exception as e:
            print(f"{n:<5} | {cov:<10} | FAILED ({str(e)})")

    best_shapes.append(local_best_shape)
    print("-" * 60)
    print(f">> WINNER for {n} dims: {local_best_shape.upper()}")
    print("-" * 60)

# Plotting the Winners
plt.figure(figsize=(8, 4))
plt.bar([str(c) for c in components_list], [1]*5, color='lightgray') # Background
plt.title("Best Covariance Shape for Each Dimension")
plt.xlabel("Number of PCA Components")
plt.yticks([])

for i, (n, shape) in enumerate(zip(components_list, best_shapes)):
    plt.text(i, 0.5, shape.upper(), ha='center', va='center', fontweight='bold', fontsize=12)

plt.show()