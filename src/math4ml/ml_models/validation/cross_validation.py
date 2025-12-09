import numpy as np

class CrossValidation:
    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True, random_seed=None):
        """
        Split data into training and testing sets.

        Parameters:
        - X (array-like): Features
        - y (array-like): Labels
        - test_size (float): Fraction of data to use as test set. Default=0.2
        - shuffle (bool): Whether to shuffle data before splitting. Default=True
        - random_seed (int or None): Random seed for reproducibility. Default=None

        Returns:
        - X_train, X_test, y_train, y_test
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]

        if shuffle:
            rng = np.random.default_rng(random_seed)
            indices = rng.permutation(n_samples)
            X = X[indices]
            y = y[indices]

        split_idx = int(n_samples * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def k_fold_split(X, y, k=5, shuffle=True, random_seed=None):
        """
        Generate K-fold splits.

        Parameters:
        - X (array-like): Features
        - y (array-like): Labels
        - k (int): Number of folds. Default=5
        - shuffle (bool): Whether to shuffle before splitting. Default=True
        - random_seed (int or None): Random seed for reproducibility. Default=None

        Returns:
        - List of tuples: [(X_train, X_val, y_train, y_val), ...] for each fold
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]

        if shuffle:
            rng = np.random.default_rng(random_seed)
            indices = rng.permutation(n_samples)
            X = X[indices]
            y = y[indices]

        fold_sizes = (n_samples // k) * np.ones(k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        current = 0
        folds = []

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            X_val, y_val = X[start:stop], y[start:stop]
            X_train = np.concatenate([X[:start], X[stop:]], axis=0)
            y_train = np.concatenate([y[:start], y[stop:]], axis=0)
            folds.append((X_train, X_val, y_train, y_val))
            current = stop

        return folds

    @staticmethod
    def shuffle_split(X, y, n_splits=5, test_size=0.2, random_seed=None):
        """
        Generate multiple random train/test splits.

        Parameters:
        - X (array-like): Features
        - y (array-like): Labels
        - n_splits (int): Number of random splits. Default=5
        - test_size (float): Fraction of data for test set. Default=0.2
        - random_seed (int or None): Random seed for reproducibility. Default=None

        Returns:
        - List of tuples: [(X_train, X_test, y_train, y_test), ...]
        """
        splits = []
        rng = np.random.default_rng(random_seed)
        n_samples = X.shape[0]

        for _ in range(n_splits):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            split_idx = int(n_samples * (1 - test_size))
            X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
            y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

    @staticmethod
    def help():
        print("CrossValidation")
        print("Methods:")
        print(" - train_test_split(X, y, test_size=0.2, shuffle=True, random_seed=None)")
        print(" - k_fold_split(X, y, k=5, shuffle=True, random_seed=None)")
        print(" - shuffle_split(X, y, n_splits=5, test_size=0.2, random_seed=None)")
