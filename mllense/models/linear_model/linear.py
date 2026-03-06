import numpy as np
from mllense.models.base import BaseEstimator
from mllense.models.core.metadata import ModelMetadata, ModelResult
from mllense.models.core.trace import Trace

class LinearRegression(BaseEstimator):
    metadata = ModelMetadata(
        name="ordinary_least_squares",
        model_type="regression",
        complexity="O(n*p^2 + p^3)",
        description="Linear Regression using Orthogonal/Singular/Normal Equation projections to minimize least squares."
    )

    def __init__(self, fit_intercept=True, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record(
            operation="fit_start",
            description=f"Initializing Linear Regression (fit_intercept={self.fit_intercept})",
            complexity_note="O(1)"
        )
        
        X = np.array(X)
        y = np.array(y)
        
        if self.fit_intercept:
            trace.record("preprocessing", "Adding a column of ones to X to compute the intercept term (bias)")
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            trace.record("preprocessing", "Skipping intercept column generation")
            X_b = X
            
        trace.record(
            operation="normal_equation", 
            description="Solving normal equations: theta = (X^T @ X)^-1 @ X^T @ y",
            complexity_note=self.metadata.complexity
        )
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        if self.fit_intercept:
            self.intercept_ = theta_best[0]
            self.coef_ = theta_best[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta_best
            
        trace.record("fit_done", f"Model fitted. Intercept: {self.intercept_:.4f}")
        
        if self.what_lense_enabled:
            self.what_lense = self._generate_what_lense()
            
        if self.how_lense_enabled:
            self.how_lense = self._finalize_how_lense(trace)
            
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Predicting dependent variable values")
        
        X = np.array(X)
        trace.record("linear_combo", "Computing y_pred = X @ coef_ + intercept_", complexity_note="O(n_samples * n_features)")
        preds = X.dot(self.coef_) + self.intercept_
        
        trace.record("predict_done", f"Generated {preds.shape[0]} predictions")
        
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        
        return ModelResult(preds, what_lense=what, how_lense=how, metadata=self.metadata)
