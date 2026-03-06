import numpy as np
from mllense.models.base import BaseEstimator
from mllense.models.core.metadata import ModelMetadata, ModelResult
from mllense.models.core.trace import Trace

class KMeans(BaseEstimator):
    metadata = ModelMetadata(
        name="k_means_clustering",
        model_type="clustering",
        complexity="O(max_iter * k * n_samples * n_features)",
        description="K-Means splits samples into k clusters by defining centroids and minimizing Euclidean distances iteratively."
    )

    def __init__(self, n_clusters=8, max_iter=300, random_state=None, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))

    def fit(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("fit_start", f"Running KMeans initialization (k={self.n_clusters})")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        X = np.array(X)
        trace.record("forgy_init", f"Selecting {self.n_clusters} random indices as initial centroids")
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]
        
        for i in range(self.max_iter):
            if i < 2:
                trace.record("iteration", f"Iter {i+1}: Computing Euclidean distance to assign points to nearest centroid")
                trace.record("centroid_shift", f"Iter {i+1}: Recalculating centroids as geometric center of respective assigned points")
            elif i == 2:
                trace.record("early_stop_poll", "... Hiding repeating iteration logs ...")

            distances = np.array([self._euclidean_distance(X, center) for center in self.cluster_centers_])
            self.labels_ = np.argmin(distances, axis=0)
            
            new_centers = np.array([
                X[self.labels_ == k].mean(axis=0) if np.any(self.labels_ == k) else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])
            
            if np.allclose(self.cluster_centers_, new_centers):
                trace.record("convergence", f"Centroids no longer moving. Converged safely after {i+1} iterations")
                break
                
            self.cluster_centers_ = new_centers
            
        trace.record("fit_done", "KMeans clustering complete.")
        
        self.what_lense = self._generate_what_lense() if self.what_lense_enabled else ""
        self.how_lense = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Predicting cluster affiliations for unobserved data")
        
        X = np.array(X)
        trace.record("evaluating_distances", "Returning argmin over Euclidean distances from each point to established cluster centers")
        
        distances = np.array([self._euclidean_distance(X, center) for center in self.cluster_centers_])
        preds = np.argmin(distances, axis=0)
        
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return ModelResult(preds, what_lense=what, how_lense=how, metadata=self.metadata)
    
    def fit_predict(self, X):
        return self.fit(X).labels_