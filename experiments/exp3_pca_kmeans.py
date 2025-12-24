import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from decomposition.pca import PCA
from clustering.kmeans import Kmeans
from utils.metrics import silhouette_score_manual, purity_score

# Load Data
print("Loading Data...")
data = load_breast_cancer()
X = data.data
y = data.target
# Normalize 
X_scaled = StandardScaler().fit_transform(X)

# I need to see how the "Shadow Size" (n_components) affects the result
components_list = [2, 5, 10, 15, 20]

purities = []
silhouettes = []
reconstruction_errors = []

print(f"Original Data Dimensions: {X_scaled.shape[1]}")
print("Starting Experiment 3...")

for n in components_list:
    print(f"\nTesting with {n} Principal Components...")
    
    # Simplify (PCA)
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    # Calculate how much info we lost (Reconstruction Error)
    error = pca.reconstruction_error(X_scaled)
    reconstruction_errors.append(error)
    print(f"  Reconstruction Error: {error:.4f}")
    
    # B. Cluster (K-Means)
    # used K=2 because I know there are 2 real answers (Benign/Malignant)
    # This lets us check if PCA kept the important info.
    kmeans = Kmeans(K=2, max_iter=100, init_method='kmeans++')
    labels = kmeans.predict(X_pca)
    
    # Score
    pur = purity_score(y, labels)
    sil = silhouette_score_manual(X_pca, labels)
    
    purities.append(pur)
    silhouettes.append(sil)
    print(f"  Purity: {pur:.4f} | Silhouette: {sil:.4f}")

# Plotting the Trade-off
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Number of PCA Components')
ax1.set_ylabel('Clustering Performance', color='tab:blue')
ax1.plot(components_list, purities, marker='o', color='tab:blue', label='Purity (Accuracy)')
ax1.plot(components_list, silhouettes, marker='x', linestyle='--', color='tab:cyan', label='Silhouette')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

ax2 = ax1.twinx() 
ax2.set_ylabel('Reconstruction Error (Lower is Better)', color='tab:red')
ax2.plot(components_list, reconstruction_errors, marker='s', color='tab:red', label='Reconstruction Error')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title("Experiment 3: PCA vs Clustering Quality")
plt.grid(True, alpha=0.3)
plt.show()