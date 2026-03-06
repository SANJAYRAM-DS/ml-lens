import numpy as np

class Utils:
    @staticmethod
    def shuffle_data(X, y, random_seed=None):
        """
        Shuffle features and labels in unison.

        Parameters:
        - X (array-like): Features
        - y (array-like): Labels
        - random_seed (int or None): Random seed for reproducibility. Default=None

        Returns:
        - X_shuffled, y_shuffled
        """
        X = np.array(X)
        y = np.array(y)
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(X.shape[0])
        return X[indices], y[indices]

    @staticmethod
    def batch_generator(X, y, batch_size=32, shuffle=True, random_seed=None):
        """
        Yield mini-batches of data for training.

        Parameters:
        - X (array-like): Features
        - y (array-like): Labels
        - batch_size (int): Size of each batch. Default=32
        - shuffle (bool): Whether to shuffle before batching. Default=True
        - random_seed (int or None): Random seed. Default=None

        Yields:
        - X_batch, y_batch
        """
        X, y = np.array(X), np.array(y)
        n_samples = X.shape[0]

        if shuffle:
            X, y = Utils.shuffle_data(X, y, random_seed=random_seed)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            yield X[start:end], y[start:end]

    @staticmethod
    def label_to_onehot(y, num_classes=None):
        """
        Convert integer labels to one-hot encoding.

        Parameters:
        - y (array-like): Labels
        - num_classes (int or None): Number of classes. If None, inferred from y.

        Returns:
        - onehot (numpy array): One-hot encoded labels
        """
        y = np.array(y).flatten()
        if num_classes is None:
            num_classes = np.max(y) + 1
        onehot = np.zeros((y.size, num_classes))
        onehot[np.arange(y.size), y] = 1
        return onehot

    @staticmethod
    def help():
        print("Utils")
        print("Methods:")
        print(" - shuffle_data(X, y, random_seed=None)")
        print(" - batch_generator(X, y, batch_size=32, shuffle=True, random_seed=None)")
        print(" - label_to_onehot(y, num_classes=None)")
