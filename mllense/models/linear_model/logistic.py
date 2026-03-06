import numpy as np
from mllense.models.base import BaseEstimator
from mllense.models.core.metadata import ModelMetadata, ModelResult
from mllense.models.core.trace import Trace

class LogisticRegression(BaseEstimator):
    metadata = ModelMetadata(
        name="logistic_regression",
        model_type="classification",
        complexity="O(max_iter * n_features * n_samples)",
        description="Logistic Regression via gradient descent mapping linear outputs to a [0, 1] probability curve using a sigmoid function."
    )

    def __init__(self, learning_rate=0.01, max_iter=1000, what_lense=False, how_lense=False):
        super().__init__(what_lense=what_lense, how_lense=how_lense)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("fit_start", f"Initializing Gradient Descent (iter={self.max_iter}, lr={self.learning_rate})")
        
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        
        self.coef_ = np.zeros(n)
        self.intercept_ = 0.0
        
        trace.record("gradient_descent", "Entering optimization loop: w = w - lr * dJ/dw, b = b - lr * dJ/db")
        for i in range(self.max_iter):
            linear_model = np.dot(X, self.coef_) + self.intercept_
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1 / m) * np.dot(X.T, (y_predicted - y))
            db = (1 / m) * np.sum(y_predicted - y)
            
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
            
            if i == 0 or i == self.max_iter - 1:
                trace.record("iteration", f"Step {i+1}: Adjusted weights based on prediction errors")
            
        trace.record("fit_done", "Logistic Regression fit complete")
        
        if self.what_lense_enabled:
            self.what_lense = self._generate_what_lense()
            
        if self.how_lense_enabled:
            self.how_lense = self._finalize_how_lense(trace)
            
        return self

    def predict(self, X):
        trace = Trace(enabled=self.how_lense_enabled)
        trace.record("predict_start", "Predicting target classes")
        
        X = np.array(X)
        linear_model = np.dot(X, self.coef_) + self.intercept_
        y_predicted = self._sigmoid(linear_model)
        
        trace.record("thresholding", "Rounding probabilities to 0/1 using a > 0.5 decision boundary")
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        
        what = self._generate_what_lense() if self.what_lense_enabled else ""
        how = self._finalize_how_lense(trace) if self.how_lense_enabled else ""
        
        return ModelResult(np.array(y_predicted_cls), what_lense=what, how_lense=how, metadata=self.metadata)
