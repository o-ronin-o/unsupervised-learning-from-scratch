import numpy as np

def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b)**2))

def silhouette_score_manual(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return 0 
    
    silhouette_vals = []
    for i in range(n_samples):
        own_cluster = labels[i]
        same_cluster_mask = (labels == own_cluster)
        if np.sum(same_cluster_mask) == 1:
            silhouette_vals.append(0)
            continue
            
        other_points_indices = np.where(same_cluster_mask)[0]
        other_points_indices = other_points_indices[other_points_indices != i]
        
        # a: Mean distance to own cluster
        a = np.mean([euclidean_dist(X[i], X[idx]) for idx in other_points_indices])
        
        # b: Mean distance to nearest other cluster
        b_candidates = []
        for label in unique_labels:
            if label == own_cluster: continue
            other_cluster_points = X[labels == label]
            mean_dist = np.mean([euclidean_dist(X[i], p) for p in other_cluster_points])
            b_candidates.append(mean_dist)
            
        b = np.min(b_candidates) if b_candidates else 0
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_vals.append(s)
        
    return np.mean(silhouette_vals)

def purity_score(y_true, y_pred):
    # Compute contingency matrix
    contingency_matrix = np.zeros((np.max(y_true)+1, np.max(int(y_pred.max()))+1))
    for t, p in zip(y_true, y_pred):
        contingency_matrix[int(t), int(p)] += 1
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)