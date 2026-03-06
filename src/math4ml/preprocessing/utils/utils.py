# preprocessing/utils/utils.py

import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

class Utils:
    """
    Utilities for preprocessing workflows.
    Includes train/test split, shuffling, and batch generation.
    """

    @staticmethod
    def split_data(X, y=None, test_size=0.2, random_state=None, help=False):
        """
        Split data into training and test sets.
        Works with X alone or X and y together.
        """
        if y is None:
            X_train, X_test = sk_train_test_split(X, test_size=test_size, random_state=random_state)
            explanation = f"train_test_split: Splitting data into {1-test_size:.0%} train and {test_size:.0%} test."
            if help:
                print(explanation)
            return X_train, X_test
        else:
            X_train, X_test, y_train, y_test = sk_train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            explanation = f"train_test_split: Splitting X and y into {1-test_size:.0%} train and {test_size:.0%} test."
            if help:
                print(explanation)
            return X_train, X_test, y_train, y_test

    @staticmethod
    def shuffle_data(X, y=None, random_state=None, help=False):
        """
        Shuffle dataset randomly.
        Works with X alone or X and y together.
        """
        rng = np.random.default_rng(seed=random_state)
        idx = rng.permutation(len(X))

        if y is None:
            X_shuffled = X[idx]
            explanation = "shuffle_data: randomly permuted X."
            if help:
                print(explanation)
            return X_shuffled
        else:
            X_shuffled, y_shuffled = X[idx], y[idx]
            explanation = "shuffle_data: randomly permuted X and y together."
            if help:
                print(explanation)
            return X_shuffled, y_shuffled

    @staticmethod
    def batch_generator(X, y=None, batch_size=32, shuffle=True):
        """
        Yield mini-batches of data for training loops.
        """
        n_samples = len(X)
        idx = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = idx[start:end]
            if y is None:
                yield X[batch_idx]
            else:
                yield X[batch_idx], y[batch_idx]
