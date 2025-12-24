import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class Kmeans:
    def __init__(self, K=5, max_iter=100, plot_steps=False, init_method='random'):
        self.K = K
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        self.init_method = init_method
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.inertia_history = [] 

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # 1. Initialize Centroids
        self.centroids = self._initialize_centroids(self.X)

        # 2. Optimization Loop
        for _ in range(self.max_iter):
            self.clusters = self._create_clusters(self.centroids)
            
            # Save inertia (Sum of squared errors)
            self.inertia_history.append(self._calculate_inertia(self.clusters, self.centroids))

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _initialize_centroids(self, X):
        """K-Means++ Initialization Logic"""
        centroids = []
        if self.init_method == 'kmeans++':
            centroids.append(X[np.random.randint(self.n_samples)])
            for _ in range(self.K - 1):
                dists = []
                for sample in X:
                    min_dist = min([euclidean_distance(sample, c) for c in centroids])
                    dists.append(min_dist**2)
                probs = dists / np.sum(dists)
                next_centroid_idx = np.random.choice(self.n_samples, p=probs)
                centroids.append(X[next_centroid_idx])
        else: 
            idxs = np.random.choice(self.n_samples, self.K, replace=False)
            centroids = [self.X[idx] for idx in idxs]
        return np.array(centroids)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._get_closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _get_closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) == 0:
                centroids[cluster_idx] = self.X[np.random.randint(self.n_samples)]
            else:
                cluster_mean = np.mean(self.X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _calculate_inertia(self, clusters, centroids):
        inertia = 0
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) > 0:
                cluster_points = self.X[cluster_indices]
                inertia += np.sum((cluster_points - centroids[i])**2)
        return inertia