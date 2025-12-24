import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from clustering.GMM import GMM
from utils.metrics import purity_score

def calculate_num_parameters(k, dim, cov_type):
    """
    Helper to count how many variables the model is learning.
    """
    # Means (k * dim) + Priors (k - 1)
    params = k * dim + (k - 1)
    
    if cov_type == 'full':
        # k * (dim * (dim+1) / 2)
        cov_params = k * (dim * (dim + 1) // 2)
    elif cov_type == 'diag':
        # k * dim
        cov_params = k * dim
    elif cov_type == 'tied':
        # One shared covariance matrix: (dim * (dim+1) / 2)
        cov_params = (dim * (dim + 1)) // 2
    elif cov_type == 'spherical':
        # k (one variance number per cluster)
        cov_params = k
        
    return params + cov_params

def calculate_bic_aic(k, X, model, cov_type):
    """
    BIC = k*ln(n) - 2*ln(L)
    AIC = 2*k - 2*ln(L)
    """
    n_samples, n_features = X.shape
    log_likelihood = model.log_likelihoods[-1] # The last value is the best one
    
    num_params = calculate_num_parameters(k, n_features, cov_type)
    
    bic = num_params * np.log(n_samples) - 2 * log_likelihood
    aic = 2 * num_params - 2 * log_likelihood
    
    return bic, aic

# 1. Load Data
print("Loading Data...")
data = load_breast_cancer()
X = data.data
y = data.target
X_scaled = StandardScaler().fit_transform(X)

# 2. Run Experiments (Test all shapes and sizes)
cov_types = ['full', 'tied', 'diag', 'spherical']
k_values = range(2, 7) # Test 2 to 6 clusters

results = {ct: {'bic': [], 'aic': []} for ct in cov_types}

print("Running GMM Experiments (This might take a moment)...")

for cov_type in cov_types:
    print(f"  Testing Shape: {cov_type}")
    for k in k_values:
        # Run GMM
        gmm = GMM(k=k, dim=X_scaled.shape[1], covariance_type=cov_type, max_iter=100)
        gmm.fit(X_scaled)
        
        # Calculate Golf Scores
        bic, aic = calculate_bic_aic(k, X_scaled, gmm, cov_type)
        
        results[cov_type]['bic'].append(bic)
        results[cov_type]['aic'].append(aic)

# 3. Plotting the Battle of Shapes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot BIC (Strict Referee)
for cov_type in cov_types:
    ax1.plot(k_values, results[cov_type]['bic'], marker='o', label=cov_type)
ax1.set_title('BIC Score (Lower is Better)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('BIC')
ax1.legend()
ax1.grid(True)

# Plot AIC (Lenient Referee)
for cov_type in cov_types:
    ax2.plot(k_values, results[cov_type]['aic'], marker='x', linestyle='--', label=cov_type)
ax2.set_title('AIC Score (Lower is Better)')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('AIC')
ax2.legend()
ax2.grid(True)

plt.show()

# Check Purity for the best theoretical model 
print("\nChecking Purity for K=2 (Full Covariance)...")
best_gmm = GMM(k=2, dim=X_scaled.shape[1], covariance_type='full')
best_gmm.fit(X_scaled)

labels = np.argmax(best_gmm.z, axis=1)
purity = purity_score(y, labels)
print(f"GMM Purity: {purity:.4f}")