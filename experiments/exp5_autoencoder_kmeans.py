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
from clustering.kmeans import Kmeans
from utils.metrics import purity_score, silhouette_score_manual

# Load Data
print("Loading Data...")
data = load_breast_cancer()
X = data.data
y = data.target
# Normalize 
X_scaled = StandardScaler().fit_transform(X)

# 2. Experiment Setup
bottleneck_sizes = [2, 5, 10, 15, 20]

results = {'dim': [], 'purity': [], 'silhouette': [], 'recon_error': []}

print("Starting Autoencoder Experiment...")
print("Note: This trains a Neural Network, so it will take longer than PCA!")

for dim in bottleneck_sizes:
    print(f"\n--- Training Autoencoder with Bottleneck={dim} ---")
    
    # Build & Train Autoencoder
    # use a small learning rate to be safe
    ae = AutoEncoder(input_dim=X_scaled.shape[1], encoding_dim=dim, 
                     learning_rate=0.001)
    
    # Train (Using 50 epochs to be quick, but 100+ is better for final results)
    ae.fit(X_scaled, epochs=50, batch_size=32, verbose=False)
    
    # Compress Data (Encode)
    X_encoded = ae.encode(X_scaled)
    
    # Calculate Reconstruction Error
    X_recon = ae.decode(X_encoded)
    error = np.mean((X_scaled - X_recon)**2)
    print(f"  Reconstruction MSE: {error:.4f}")
    
    # Cluster (K-Means)
    km = Kmeans(K=2, max_iter=100, init_method='kmeans++')
    labels = km.predict(X_encoded)
    
    # Score
    pur = purity_score(y, labels)
    sil = silhouette_score_manual(X_encoded, labels)
    print(f"  Purity: {pur:.4f} | Silhouette: {sil:.4f}")
    
    # Save results
    results['dim'].append(dim)
    results['purity'].append(pur)
    results['silhouette'].append(sil)
    results['recon_error'].append(error)

# Visualization
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Bottleneck Dimension')
ax1.set_ylabel('Purity (Higher is Better)', color='tab:blue')
ax1.plot(results['dim'], results['purity'], 'o-', color='tab:blue', label='Purity')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Reconstruction Error (Lower is Better)', color='tab:red')
ax2.plot(results['dim'], results['recon_error'], 's--', color='tab:red', label='Reconstruction Error')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title("Experiment 5: Autoencoder + K-Means")
plt.grid(True, alpha=0.3)
plt.show()

# Plot the training history of the last model just to check convergence
print("Displaying training curve for the last model...")
ae.plot_training_history()