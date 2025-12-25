import numpy as np

def purity_score(y_true, y_pred):
    """
    Calculate the purity score for the given cluster assignments and ground truth labels.
    """
    # Create a confusion matrix
    contingency_matrix = np.zeros((np.max(y_true) + 1, np.max(y_pred) + 1))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    
    # Return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def silhouette_score_manual(X, labels):
    """
    Compute the mean Silhouette Coefficient of all samples.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return 0.0
        
    silhouette_vals = np.zeros(n_samples)
    
    # Precompute distances (can be slow for large N, but okay for this dataset)
    # Using squared Euclidean for speed, then sqrt
    distances = np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))
    
    for i in range(n_samples):
        # Mean distance to other points in the same cluster
        own_cluster = labels[i]
        mask_same = (labels == own_cluster)
        mask_same[i] = False # Exclude self
        
        if np.sum(mask_same) == 0:
            a_i = 0
        else:
            a_i = np.mean(distances[i, mask_same])
            
        # Mean distance to points in the nearest OTHER cluster
        b_i = np.inf
        for label in unique_labels:
            if label == own_cluster:
                continue
            mask_other = (labels == label)
            mean_dist_other = np.mean(distances[i, mask_other])
            b_i = min(b_i, mean_dist_other)
            
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        
    return np.mean(silhouette_vals)

# --- Davies-Bouldin Index ---
def davies_bouldin_score(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2: return np.inf
    
    centroids = np.zeros((n_clusters, X.shape[1]))
    avg_dists = np.zeros(n_clusters)
    
    # 1. Calculate centroids and average distances (dispersion)
    for idx, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        centroids[idx] = np.mean(cluster_points, axis=0)
        # Average Euclidean distance from centroid
        avg_dists[idx] = np.mean(np.linalg.norm(cluster_points - centroids[idx], axis=1))
        
    score = 0
    # 2. Find max similarity for each cluster
    for i in range(n_clusters):
        max_val = -np.inf
        for j in range(n_clusters):
            if i != j:
                # Similarity: (scatter_i + scatter_j) / dist(centroid_i, centroid_j)
                dist_centroids = np.linalg.norm(centroids[i] - centroids[j])
                val = (avg_dists[i] + avg_dists[j]) / dist_centroids
                if val > max_val:
                    max_val = val
        score += max_val
        
    return score / n_clusters

# --- Calinski-Harabasz Index ---
def calinski_harabasz_score(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2: return 0.0
    
    # Global mean
    mean_global = np.mean(X, axis=0)
    
    # Between-cluster dispersion (SS_B)
    SS_B = 0
    SS_W = 0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_points = cluster_points.shape[0]
        centroid = np.mean(cluster_points, axis=0)
        
        # Weighted squared distance from global mean
        SS_B += n_points * np.sum((centroid - mean_global) ** 2)
        
        # Within-cluster dispersion (SS_W)
        SS_W += np.sum((cluster_points - centroid) ** 2)
        
    if SS_W == 0: return np.inf
    
    return (SS_B / (n_clusters - 1)) / (SS_W / (n_samples - n_clusters))

# --- NEW: Adjusted Rand Index ---
def adjusted_rand_score(labels_true, labels_pred):
    # Use the contingency matrix logic
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    # Contingency matrix
    contingency = np.zeros((len(classes), len(clusters)))
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
            
    # Sums
    sum_rows = np.sum(contingency, axis=1)
    sum_cols = np.sum(contingency, axis=0)
    n = len(labels_true)
    
    # Helper for "n choose 2"
    def comb2(x):
        return x * (x - 1) / 2
        
    sum_nij_comb = np.sum([comb2(n_ij) for n_ij in contingency.flatten()])
    sum_a_comb = np.sum([comb2(a) for a in sum_rows])
    sum_b_comb = np.sum([comb2(b) for b in sum_cols])
    
    expected_index = (sum_a_comb * sum_b_comb) / comb2(n)
    max_index = (sum_a_comb + sum_b_comb) / 2
    
    if max_index == expected_index:
        return 1.0
        
    return (sum_nij_comb - expected_index) / (max_index - expected_index)

# --- NEW: Normalized Mutual Information ---
def normalized_mutual_info_score(labels_true, labels_pred):
    # Entropy calculation
    def entropy(labels):
        n = len(labels)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log(probs + 1e-10)) # Small epsilon for stability
        
    H_true = entropy(labels_true)
    H_pred = entropy(labels_pred)
    
    # Mutual Information
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    n = len(labels_true)
    MI = 0
    
    for c in classes:
        for k in clusters:
            mask_c = (labels_true == c)
            mask_k = (labels_pred == k)
            intersect = np.sum(mask_c & mask_k)
            if intersect > 0:
                p_ck = intersect / n
                p_c = np.sum(mask_c) / n
                p_k = np.sum(mask_k) / n
                MI += p_ck * np.log(p_ck / (p_c * p_k))
                
    return 2 * MI / (H_true + H_pred)

def confusion_matrix_manual(y_true, y_pred):
    # Ensure integer labels
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Get the number of unique classes/clusters
    n_classes = np.max(y_true) + 1
    n_clusters = np.max(y_pred) + 1
    
    # Initialize matrix
    cm = np.zeros((n_classes, n_clusters), dtype=int)
    
    # Fill matrix
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    return cm
    
def calculate_aic(bic, n_samples):
    # Quick helper: AIC = BIC - log(n)*k + 2*k
    # But strictly: AIC = 2k - 2ln(L)
    # Since we have BIC = k*ln(n) - 2ln(L)
    # We can't perfectly convert without k, but we can compute it inside the notebook using the log-likelihood
    pass