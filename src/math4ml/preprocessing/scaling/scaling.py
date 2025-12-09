# preprocessing/scaling/scaling.py

import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler

class Scaling:
    """
    Scaling class for numerical features.
    Provides static methods for Standard, MinMax, MaxAbs, and Robust scaling.
    """

    @staticmethod
    def standard(X, help=False):
        """
        StandardScaler: z-score normalization (mean=0, std=1)
        """
        scaler = SklearnStandardScaler()
        X_scaled = scaler.fit_transform(X)
        if help:
            print("StandardScaler: z-score normalization (mean=0, std=1).")
            print(f"Input shape: {X.shape}, Output shape: {X_scaled.shape}")
        return X_scaled

    @staticmethod
    def minmax(X, feature_range=(0,1), help=False):
        """
        MinMaxScaler: scales features to given range (default 0-1)
        """
        scaler = SklearnMinMaxScaler(feature_range=feature_range)
        X_scaled = scaler.fit_transform(X)
        if help:
            print(f"MinMaxScaler: scales features to range {feature_range}.")
            print(f"Input shape: {X.shape}, Output shape: {X_scaled.shape}")
        return X_scaled

    @staticmethod
    def maxabs(X, help=False):
        """
        MaxAbsScaler: scales features to [-1, 1] by max absolute value
        """
        scaler = SklearnMaxAbsScaler()
        X_scaled = scaler.fit_transform(X)
        if help:
            print("MaxAbsScaler: scales features to [-1,1] by max absolute value.")
            print(f"Input shape: {X.shape}, Output shape: {X_scaled.shape}")
        return X_scaled

    @staticmethod
    def robust(X, help=False):
        """
        RobustScaler: scales features using median & IQR (robust to outliers)
        """
        scaler = SklearnRobustScaler()
        X_scaled = scaler.fit_transform(X)
        if help:
            print("RobustScaler: scales features using median and interquartile range (robust to outliers).")
            print(f"Input shape: {X.shape}, Output shape: {X_scaled.shape}")
        return X_scaled
