import numpy as np
from clustering.kmeans import Kmeans
from clustering.GMM import GMM

def calculate_gap_statistic(X, k_max=7, n_refs=5):
    """
    Calculates the Gap Statistic for K-Means to find optimal k.
    
    Args:
        X: Dataset (n_samples, n_features)
        k_max: Maximum k to test (default 7)
        n_refs: Number of random reference datasets to generate (default 5)
        
    Returns:
        k_values: range of k tested
        gaps: Gap values for each k (Higher is better)
    """
    gaps = []
    k_values = range(2, k_max + 1)
    
    # Helper to calculate log(W_k)
    def get_log_wk(data, k):
        km = Kmeans(K=k, max_iter=50, init_method='kmeans++')
        km.predict(data)
        return np.log(km.inertia_history[-1] + 1e-10)

    print(f"Calculating Gap Statistic (Testing K=2 to {k_max})...")
    
    for k in k_values:
        # 1. Log(Wk) for original data
        log_wk = get_log_wk(X, k)
        
        # 2. Log(Wk) for reference datasets
        ref_log_wks = []
        for _ in range(n_refs):
            X_ref = np.random.uniform(
                low=X.min(axis=0), 
                high=X.max(axis=0), 
                size=X.shape
            )
            ref_log_wks.append(get_log_wk(X_ref, k))
            
        gap = np.mean(ref_log_wks) - log_wk
        gaps.append(gap)
        
    return k_values, gaps

def run_gmm_grid_search(X, k_values=range(2, 7), cov_types=None):
    
    if cov_types is None:
        cov_types = ['full', 'tied', 'diag', 'spherical']
        
    results = []
    best_bic = float('inf')
    best_config = None
    best_model = None
    n, d = X.shape

    print("Running GMM Grid Search...")

    for cov in cov_types:
        bic_scores = []
        aic_scores = []
        
        for k in k_values:
            try:
                gmm = GMM(k=k, dim=d, covariance_type=cov, max_iter=100)
                gmm.fit(X)
                
                # Metrics Calculation
                ll = gmm.log_likelihoods[-1]
                
                # Count Parameters
                if cov == 'full': n_params = k*d + k*(d*(d+1)//2) + (k-1)
                elif cov == 'diag': n_params = k*d + k*d + (k-1)
                elif cov == 'tied': n_params = k*d + (d*(d+1)//2) + (k-1)
                else: n_params = k*d + k + (k-1) # spherical
                    
                bic = n_params * np.log(n) - 2 * ll
                aic = 2 * n_params - 2 * ll
                
                bic_scores.append(bic)
                aic_scores.append(aic)
                
                # Check for winner
                if bic < best_bic:
                    best_bic = bic
                    best_config = (cov, k)
                    best_model = gmm
                    
            except Exception as e:
                # Handle singular matrix errors
                bic_scores.append(np.nan)
                aic_scores.append(np.nan)
        
        results.append({
            'cov': cov,
            'bic': bic_scores,
            'aic': aic_scores
        })
        
    return results, best_model, best_config