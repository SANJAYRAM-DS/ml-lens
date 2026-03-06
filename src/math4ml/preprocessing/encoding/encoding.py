# preprocessing/encoding/encoding.py

import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder

class Encoding:
    """
    Encoding class for categorical variables.
    Provides static methods for OneHot, Label, and Ordinal encoding.
    All methods support 'help=True' for explanation.
    """

    @staticmethod
    def onehot(X, help=False, sparse=False):
        """
        OneHot encode categorical features.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
            help : bool, default=False
                If True, prints explanation
            sparse : bool, default=False
                If True, returns a sparse matrix
            
        Returns:
            X_encoded : ndarray or sparse matrix
        """
        encoder = SklearnOneHotEncoder(sparse=sparse)
        X_encoded = encoder.fit_transform(X)
        if help:
            print("OneHotEncoder: converts categorical features into binary vectors.")
            print(f"Input shape: {X.shape}, Output shape: {X_encoded.shape}")
        return X_encoded

    @staticmethod
    def label(X, help=False):
        """
        Label encode categorical features (integer labels).
        Works column-wise if X has multiple features.
        
        Parameters:
            X : array-like of shape (n_samples,) or (n_samples, n_features)
            help : bool, default=False
                If True, prints explanation
        
        Returns:
            X_encoded : ndarray of integer labels
        """
        X = np.array(X)
        if X.ndim == 1:
            encoder = SklearnLabelEncoder()
            X_encoded = encoder.fit_transform(X)
        else:
            # Apply LabelEncoder to each column
            X_encoded = np.zeros_like(X, dtype=int)
            for i in range(X.shape[1]):
                encoder = SklearnLabelEncoder()
                X_encoded[:, i] = encoder.fit_transform(X[:, i])
        if help:
            print("LabelEncoder: converts categorical values into integer labels.")
            print(f"Input shape: {X.shape}, Output shape: {X_encoded.shape}")
        return X_encoded

    @staticmethod
    def ordinal(X, help=False):
        """
        Ordinal encode categorical features (ordered integers).
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
            help : bool, default=False
                If True, prints explanation
        
        Returns:
            X_encoded : ndarray of integers
        """
        X = np.array(X)
        encoder = SklearnOrdinalEncoder()
        X_encoded = encoder.fit_transform(X)
        if help:
            print("OrdinalEncoder: encodes categories with ordered integers.")
            print(f"Input shape: {X.shape}, Output shape: {X_encoded.shape}")
        return X_encoded
