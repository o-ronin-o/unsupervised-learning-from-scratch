import numpy as np 
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class Kmeans:

    def __init__(self, K = 5, max_iter=100,plot_steps =False):
        self.K = K 
        self.max_iter = max_iter
        self.plot_steps= plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

        def predict(self, X):
            self.X = X
            self.n_samples, self.n_features =  X.shape
            
            # take random samples and treat them as centroids 
            idxs = np.random.choice(self.n_samples, self.K, replace = False)
            self.centroids  = [self.X[idx] for idx in idxs]

            # optimization loop 
            for _ in range(self.max_iter):
                # first we assign samples to the closest centroid (create clusters)
                self.clusters = self._create_clusters(self.centroids)

                #then we update our centroids according to our new created clusters actual mean
                centroids_old = self.centroids # we use this to track convergence 
                self.centroids = self._get_centroids(self.clusters)

                if self._is_converged(centroids_old,self.centroids):
                    break
                # if self.plot_steps:
                #     self.plot()


            # now we classify samples
            return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        # to get  cluster labels we will assign cluster label to each sample it includes
        # that is done with the assumption the _create_clusters already clustered things

        #initialize labels array
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # we need to assign the samples to the closest centroid 
        # first we initialize clusters 
        clusters =  [[] for _ in range(self.K)]
        
        # now we "in simple words" grab the sample along with it's idx
        # then we pass it to helper function to determine the suitable cluster 
        for idx, sample in enumerate(self.X):
            centroid_idx = self._get_closest_centroid(sample,centroids)
            # now we put the index in the suiable cluster
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _get_closest_centroid(self, sample, centroids):
        # we use euclidean distance 
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx , cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    def _is_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0 
    