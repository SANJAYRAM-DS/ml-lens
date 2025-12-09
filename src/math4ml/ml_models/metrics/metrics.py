import numpy as np
from collections import Counter

# -----------------------------
# Regression Metrics
# -----------------------------
class RegressionMetrics:
    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    @staticmethod
    def help():
        print("RegressionMetrics")
        print("Methods:")
        print(" - mse(y_true, y_pred): Mean Squared Error")
        print(" - rmse(y_true, y_pred): Root Mean Squared Error")
        print(" - mae(y_true, y_pred): Mean Absolute Error")
        print(" - r2(y_true, y_pred): R-squared (coefficient of determination)")


# -----------------------------
# Classification Metrics
# -----------------------------
class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred, pos_label=1):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return tp / (tp + fp + 1e-9)

    @staticmethod
    def recall(y_true, y_pred, pos_label=1):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return tp / (tp + fn + 1e-9)

    @staticmethod
    def f1_score(y_true, y_pred, pos_label=1):
        p = ClassificationMetrics.precision(y_true, y_pred, pos_label)
        r = ClassificationMetrics.recall(y_true, y_pred, pos_label)
        return 2 * p * r / (p + r + 1e-9)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        for true, pred in zip(y_true, y_pred):
            i = label_to_index[true]
            j = label_to_index[pred]
            matrix[i, j] += 1
        return matrix

    @staticmethod
    def help():
        print("ClassificationMetrics")
        print("Methods:")
        print(" - accuracy(y_true, y_pred)")
        print(" - precision(y_true, y_pred, pos_label=1)")
        print(" - recall(y_true, y_pred, pos_label=1)")
        print(" - f1_score(y_true, y_pred, pos_label=1)")
        print(" - confusion_matrix(y_true, y_pred)")
