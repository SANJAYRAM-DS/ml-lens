# preprocessing/feature_engineering/feature_engineering.py

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer

class FeatureEngineering:
    """
    FeatureEngineering class for generating new features.
    Includes polynomial features, interaction terms, log and power transforms.
    """

    @staticmethod
    def polynomial(X, degree=2, include_bias=False, help=False):
        """
        Generate polynomial features up to given degree.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
            degree : int, default=2
                Degree of polynomial features
            include_bias : bool, default=False
                If True, includes bias (column of ones)
            help : bool, default=False
                If True, prints explanation
        
        Returns:
            X_poly : ndarray of shape (n_samples, n_output_features)
        """
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        if help:
            print(f"PolynomialFeatures: generates polynomial features up to degree={degree}.")
            print(f"Input shape: {X.shape}, Output shape: {X_poly.shape}")
        return X_poly

    @staticmethod
    def interaction(X, include_bias=False, help=False):
        """
        Generate interaction terms between features (pairwise products).
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
            include_bias : bool, default=False
                If True, includes bias column
            help : bool, default=False
                If True, prints explanation
        
        Returns:
            X_inter : ndarray with interaction terms
        """
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=include_bias)
        X_inter = poly.fit_transform(X)
        if help:
            print("InteractionTerms: generates pairwise interaction features (x_i * x_j).")
            print(f"Input shape: {X.shape}, Output shape: {X_inter.shape}")
        return X_inter

    @staticmethod
    def log_transform(X, help=False):
        """
        Apply log transform to features (log(1 + x) to avoid log(0)).
        
        Parameters:
            X : array-like, must be non-negative
            help : bool, default=False
        
        Returns:
            X_log : ndarray
        """
        X = np.array(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("Log transform requires non-negative values.")
        X_log = np.log1p(X)
        if help:
            print("LogTransform: applies log(1 + x) to stabilize variance.")
            print(f"Input shape: {X.shape}, Output shape: {X_log.shape}")
        return X_log

    @staticmethod
    def power_transform(X, method='yeo-johnson', help=False):
        """
        Apply power transform to make features more Gaussian-like.
        
        Parameters:
            X : array-like
            method : 'yeo-johnson' (supports negative) or 'box-cox' (positive only)
            help : bool, default=False
        
        Returns:
            X_trans : ndarray
        """
        pt = PowerTransformer(method=method)
        X_trans = pt.fit_transform(X)
        if help:
            print(f"PowerTransform: {method} transformation to stabilize variance & make data more Gaussian-like.")
            print(f"Input shape: {X.shape}, Output shape: {X_trans.shape}")
        return X_trans
