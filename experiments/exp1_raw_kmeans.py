import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Looking for files in the folder one level up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --------------------------------------------------

from clustering.kmeans import Kmeans
from utils.metrics import silhouette_score_manual, purity_score

# Load Data
print("Loading Breast Cancer Data...")
data = load_breast_cancer()
X = data.data
y = data.target
# Normalize
X_scaled = StandardScaler().fit_transform(X)

# Elbow Method (Try K=2 to K=7)
print("Running Elbow Method...")
k_values = range(2, 8)
inertias = []
sil_scores = []

for k in k_values:
    print(f"  Testing K={k}...")
    # Use Kmeans++ implementation
    km = Kmeans(K=k, max_iter=100, init_method='kmeans++')
    labels = km.predict(X_scaled)
    
    inertias.append(km.inertia_history[-1])
    sil_scores.append(silhouette_score_manual(X_scaled, labels))

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Lower is Better)', color='tab:red')
ax1.plot(k_values, inertias, 'o-', color='tab:red', label='Inertia')

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score (Higher is Better)', color='tab:blue')
ax2.plot(k_values, sil_scores, 'x--', color='tab:blue', label='Silhouette')

plt.title("Experiment 1: Elbow Method & Silhouette Analysis")
plt.show()

# Final Comparison (Random vs K-Means++)
print("\nComparing Random Init vs K-Means++ (K=2)...")
km_rand = Kmeans(K=2, init_method='random')
purity_rand = purity_score(y, km_rand.predict(X_scaled))

km_pp = Kmeans(K=2, init_method='kmeans++')
purity_pp = purity_score(y, km_pp.predict(X_scaled))

print(f"Random Purity:   {purity_rand:.4f}")
print(f"KMeans++ Purity: {purity_pp:.4f}")