import numpy as np
from collections import Counter

# -----------------------------
# Logistic Regression
# -----------------------------
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, fit_intercept=True):
        """
        Logistic Regression for binary classification

        Parameters:
        - lr (float): Learning rate for gradient descent. Default=0.01
        - epochs (int): Number of iterations. Default=1000
        - fit_intercept (bool): Whether to include intercept term. Default=True
        """
        self.lr = lr
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])

        beta = np.zeros((X.shape[1], 1))

        for _ in range(self.epochs):
            predictions = self._sigmoid(X @ beta)
            gradient = X.T @ (predictions - y) / n_samples
            beta -= self.lr * gradient

        if self.fit_intercept:
            self.intercept_ = beta[0, 0]
            self.coef_ = beta[1:].flatten()
        else:
            self.intercept_ = 0
            self.coef_ = beta.flatten()

    def predict_proba(self, X):
        X = np.array(X)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        z = X @ np.hstack([[self.intercept_], self.coef_])
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def help(self):
        print("LogisticRegression(lr=0.01, epochs=1000, fit_intercept=True)")
        print("Methods:")
        print(" - fit(X, y): Fit the model to data")
        print(" - predict(X, threshold=0.5): Predict class labels")
        print(" - predict_proba(X): Predict probabilities")
        print("Attributes:")
        print(" - coef_: Coefficients of features")
        print(" - intercept_: Intercept term")


# -----------------------------
# Naive Bayes
# -----------------------------
class NaiveBayes:
    def __init__(self, model_type="gaussian"):
        """
        Naive Bayes classifier

        Parameters:
        - model_type (str): "gaussian" or "multinomial". Default="gaussian"
        """
        self.model_type = model_type
        self.classes_ = None
        self.params_ = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        if self.model_type == "gaussian":
            self.params_ = {}
            for c in self.classes_:
                X_c = X[y == c]
                self.params_[c] = {
                    "mean": X_c.mean(axis=0),
                    "var": X_c.var(axis=0) + 1e-9,
                    "prior": X_c.shape[0] / X.shape[0]
                }
        elif self.model_type == "multinomial":
            self.params_ = {}
            X = X.astype(int)
            for c in self.classes_:
                X_c = X[y == c]
                self.params_[c] = {
                    "prob": (X_c.sum(axis=0) + 1) / (X_c.sum() + X.shape[1]),
                    "prior": X_c.shape[0] / X.shape[0]
                }

    def _predict_single(self, x):
        probs = {}
        for c in self.classes_:
            if self.model_type == "gaussian":
                mean = self.params_[c]["mean"]
                var = self.params_[c]["var"]
                prior = self.params_[c]["prior"]
                # Gaussian likelihood
                likelihood = np.prod(1/np.sqrt(2*np.pi*var) * np.exp(-(x-mean)**2/(2*var)))
                probs[c] = likelihood * prior
            elif self.model_type == "multinomial":
                prob = self.params_[c]["prob"]
                prior = self.params_[c]["prior"]
                likelihood = np.prod(prob**x)
                probs[c] = likelihood * prior
        return max(probs, key=probs.get)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def help(self):
        print('NaiveBayes(model_type="gaussian")')
        print("Methods:")
        print(" - fit(X, y): Fit model to data")
        print(" - predict(X): Predict class labels")
        print("Parameters:")
        print(" - model_type: 'gaussian' or 'multinomial'")


# -----------------------------
# k-Nearest Neighbors (kNN)
# -----------------------------
class KNN:
    def __init__(self, k=3):
        """
        k-Nearest Neighbors classifier

        Parameters:
        - k (int): Number of neighbors to consider. Default=3
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            neighbors_idx = np.argsort(distances)[:self.k]
            neighbors_labels = self.y_train[neighbors_idx]
            most_common = Counter(neighbors_labels).most_common(1)[0][0]
            y_pred.append(most_common)
        return np.array(y_pred)

    def help(self):
        print("KNN(k=3)")
        print("Methods:")
        print(" - fit(X, y): Store training data")
        print(" - predict(X): Predict class labels using majority vote of k nearest neighbors")
        print("Parameters:")
        print(" - k: number of neighbors")
