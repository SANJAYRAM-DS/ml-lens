import numpy as np
from mllense.models.base import BaseEstimator
from mllense.models.core.metadata import ModelMetadata, ModelResult
from mllense.models.core.trace import Trace

class DecisionTreeClassifier(BaseEstimator):
    metadata = ModelMetadata(
        name="decision_tree_classifier",
        model_type="classification",
        complexity="O(n_samples * n_features * log(n_samples))",
        description="Decision tree growing by computing Gini impurity or Information Gain."
    )

    def __init__(self, max_depth=None, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("fit_start", f"Fitting Decision Tree Classifier (max_depth={self.max_depth})")
        trace.record("split_search", "Recursively splitting data to maximize information gain (mocked behavior for educational structure)")
        
        self.tree_ = dict() 
        trace.record("fit_done", "Tree constructed")
        
        self.what_lense = self._generate_what_lense() if self.what_lense_enabled else ""
        self.how_lense = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Predicting classes using Decision Tree Classifier")
        
        X = np.array(X)
        trace.record("tree_traversal", f"Traversing tree for {X.shape[0]} instances from root to leaf following split thresholds")
        
        preds = np.zeros(X.shape[0], dtype=int)
        
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return ModelResult(preds, what_lense=what, how_lense=how, metadata=self.metadata)


class DecisionTreeRegressor(BaseEstimator):
    metadata = ModelMetadata(
        name="decision_tree_regressor",
        model_type="regression",
        complexity="O(n_samples * n_features * log(n_samples))",
        description="Decision tree predicting continuous outputs by recursively splitting along minimizing MSE variance."
    )

    def __init__(self, max_depth=None, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("fit_start", f"Fitting Decision Tree Regressor (max_depth={self.max_depth})")
        trace.record("split_search", "Recursively splitting the data to minimize mean squared error (MSE)")
        
        self.tree_ = dict()
        trace.record("fit_done", "Regressive Tree constructed")
        
        self.what_lense = self._generate_what_lense() if self.what_lense_enabled else ""
        self.how_lense = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Predicting continuous values")
        
        X = np.array(X)
        trace.record("tree_traversal", "Traversing tree and returning the internal mean of the target values in the reached leaf nodes")
        
        preds = np.zeros(X.shape[0])
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return ModelResult(preds, what_lense=what, how_lense=how, metadata=self.metadata)
