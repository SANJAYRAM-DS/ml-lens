import numpy as np

# -----------------------------
# Linear Regression (Ordinary Least Squares)
# -----------------------------
class LinearRegression:
    def __init__(self, fit_intercept=True):
        """
        Linear Regression (OLS)
        
        Parameters:
        - fit_intercept (bool): Whether to calculate the intercept term. Default is True.
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Closed-form solution: beta = (X^T X)^-1 X^T y
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y

        if self.fit_intercept:
            self.intercept_ = beta[0, 0]
            self.coef_ = beta[1:].flatten()
        else:
            self.intercept_ = 0
            self.coef_ = beta.flatten()

    def predict(self, X):
        X = np.array(X)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def help(self):
        print("LinearRegression(fit_intercept=True)")
        print("Methods:")
        print(" - fit(X, y): Fit the model to data")
        print(" - predict(X): Predict using the fitted model")
        print("Attributes:")
        print(" - coef_: Coefficients of features")
        print(" - intercept_: Intercept term")


# -----------------------------
# Ridge Regression (L2 Regularization)
# -----------------------------
class RidgeRegression:
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        Ridge Regression (L2 Regularization)

        Parameters:
        - alpha (float): Regularization strength (λ). Default=1.0
        - fit_intercept (bool): Whether to calculate the intercept term. Default is True.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_features = X.shape[1]

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        I = np.eye(X.shape[1])
        if self.fit_intercept:
            I[0, 0] = 0  # Don't regularize intercept

        beta = np.linalg.pinv(X.T @ X + self.alpha * I) @ X.T @ y

        if self.fit_intercept:
            self.intercept_ = beta[0, 0]
            self.coef_ = beta[1:].flatten()
        else:
            self.intercept_ = 0
            self.coef_ = beta.flatten()

    def predict(self, X):
        X = np.array(X)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def help(self):
        print("RidgeRegression(alpha=1.0, fit_intercept=True)")
        print("Methods:")
        print(" - fit(X, y): Fit the model to data with L2 regularization")
        print(" - predict(X): Predict using the fitted model")
        print("Attributes:")
        print(" - coef_: Coefficients of features")
        print(" - intercept_: Intercept term")


# -----------------------------
# Lasso Regression (L1 Regularization)
# -----------------------------
class LassoRegression:
    def __init__(self, alpha=1.0, fit_intercept=True, lr=0.001, epochs=1000):
        """
        Lasso Regression (L1 Regularization) using simple Gradient Descent
        
        Parameters:
        - alpha (float): Regularization strength (λ). Default=1.0
        - fit_intercept (bool): Whether to calculate the intercept term. Default is True
        - lr (float): Learning rate for gradient descent. Default=0.001
        - epochs (int): Number of iterations for gradient descent. Default=1000
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])
        
        beta = np.zeros((X.shape[1], 1))

        # Gradient Descent
        for _ in range(self.epochs):
            prediction = X @ beta
            gradient = (X.T @ (prediction - y)) / n_samples
            # L1 penalty (subgradient)
            gradient += self.alpha * np.sign(beta)
            beta -= self.lr * gradient

        if self.fit_intercept:
            self.intercept_ = beta[0, 0]
            self.coef_ = beta[1:].flatten()
        else:
            self.intercept_ = 0
            self.coef_ = beta.flatten()

    def predict(self, X):
        X = np.array(X)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def help(self):
        print("LassoRegression(alpha=1.0, fit_intercept=True, lr=0.001, epochs=1000)")
        print("Methods:")
        print(" - fit(X, y): Fit the model to data using gradient descent with L1 penalty")
        print(" - predict(X): Predict using the fitted model")
        print("Attributes:")
        print(" - coef_: Coefficients of features")
        print(" - intercept_: Intercept term")
