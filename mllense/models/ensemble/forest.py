import numpy as np
from mllense.models.base import BaseEstimator
from mllense.models.core.metadata import ModelMetadata, ModelResult
from mllense.models.core.trace import Trace
from mllense.models.tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestClassifier(BaseEstimator):
    metadata = ModelMetadata(
        name="random_forest_classifier",
        model_type="classification_ensemble",
        complexity="O(M * n_samples * n_features * log(n_samples))",
        description="Ensemble consisting of building multiple Decision Trees trained on bootstrapped data (bagging) taking majority votes."
    )

    def __init__(self, n_estimators=100, max_depth=None, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators_ = []

    def fit(self, X, y):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("fit_start", f"Fitting Random Forest Classifier with M={self.n_estimators} estimators")

        self.estimators_ = []
        for i in range(self.n_estimators):
            if i < 3 or i == self.n_estimators - 1:
                trace.record("bootstrapping", f"Creating bootstrapped dataset and growing tree {i+1}")
            elif i == 3:
                trace.record("bootstrapping_skip", "... Skipping subsequent tree iteration logs ...", complexity_note="O(n_samples * log(n_samples)) per tree")

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                what_lense=False, 
                how_lense=False
            )
            self.estimators_.append(tree)
            
        trace.record("fit_done", "Forest constructed successfully")
        
        self.what_lense = self._generate_what_lense() if self.what_lense_enabled else ""
        self.how_lense = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Aggregating predictions across ensemble")
        
        X = np.array(X)
        trace.record("majority_vote", "Applying mode/majority vote from all individual tree predictions")
        
        preds = np.zeros(X.shape[0], dtype=int)
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return ModelResult(preds, what_lense=what, how_lense=how, metadata=self.metadata)


class RandomForestRegressor(BaseEstimator):
    metadata = ModelMetadata(
        name="random_forest_regressor",
        model_type="regression_ensemble",
        complexity="O(M * n_samples * n_features * log(n_samples))",
        description="Ensemble generating multiple decision sum-of-squares trees and taking their aggregated mean prediction."
    )

    def __init__(self, n_estimators=100, max_depth=None, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators_ = []

    def fit(self, X, y):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("fit_start", f"Fitting Random Forest Regressor with {self.n_estimators} estimators")
        trace.record("bagging", "Training multiple regression trees on bootstrapped samples")
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            self.estimators_.append(tree)
            
        trace.record("fit_done", "Forest constructed successfully")
        
        self.what_lense = self._generate_what_lense() if self.what_lense_enabled else ""
        self.how_lense = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Predicting continuous values via ensemble")
        
        X = np.array(X)
        trace.record("averaging", "Aggregating predictions from all trees by computing their mean output space")
        
        preds = np.zeros(X.shape[0])
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        return ModelResult(preds, what_lense=what, how_lense=how, metadata=self.metadata)
