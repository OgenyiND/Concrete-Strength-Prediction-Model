# utilities.py
import numpy as np

def feature_normalization(X):
    """
    Perform z-score normalization on input features.

    Parameters:
    X : numpy array, shape (n_samples, n_features)
        Input features to normalize.

    Returns:
    X_normalized : numpy array, shape (n_samples, n_features)
        Normalized input features.
    mean : numpy array, shape (n_features,)
        Mean of each feature.
    std : numpy array, shape (n_features,)
        Standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def compute_cost(X, y, w, b):
    """
    Compute the cost (mean squared error) for linear regression.
    """
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost


def compute_gradient(X, y, w, b):
    """
    Compute gradients for weights and bias in linear regression.
    """
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    errors = predictions - y
    dj_dw = (1 / m) * np.dot(X.T, errors)
    dj_db = (1 / m) * np.sum(errors)
    return dj_dw, dj_db
