# unsupervised-learning-from-scratch
A research-style implementation of unsupervised learning pipelines (PCA, Autoencoders, K-Means, GMM) from scratch using NumPy, with extensive statistical evaluation and comparison.


## Why this repo?
Most ML repos use scikit-learn. This project reimplements core
unsupervised learning algorithms from first principles to deeply
understand optimization, EM, and representation learning.

## Implemented from Scratch
- PCA (eigen-decomposition, reconstruction, variance analysis)
- Deep Autoencoder (manual backprop, LR scheduling, regularization)
- K-Means (K-Means++, convergence tracking)
- Gaussian Mixture Models (full EM, all covariance types)
- Clustering & validation metrics (internal + external)

## Experimental Pipelines
1. Raw → K-Means
2. Raw → GMM
3. PCA → K-Means
4. PCA → GMM
5. Autoencoder → K-Means
6. Autoencoder → GMM

## Key Findings
- PCA stabilizes GMM covariance estimation at low dimensions
- Autoencoders outperform PCA at moderate bottleneck sizes
- K-Means++ converges faster and more reliably than random init

## Reproducibility
- Fixed random seeds
- Deterministic NumPy implementations


## File Structure

src/
 ├── decomposition/
 │    ├── pca.py
 │    └── autoencoder.py
 ├── clustering/
 │    ├── kmeans.py
 │    └── gmm.py
 ├── metrics/
 │    ├── internal.py
 │    └── external.py
 ├── experiments/
 │    ├── exp1_raw_kmeans.py
 │    ├── exp2_raw_gmm.py
 │    └── ...
 └── utils/
      ├── math.py
      └── data.py
