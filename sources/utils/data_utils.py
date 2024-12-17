import numpy as np

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the data using min-max scaling, applied column-wise.

    Parameters:
        data (np.ndarray): The data to normalize.
    Returns:
        np.ndarray: The normalized data.
    """
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def sigmoid_data(data: np.ndarray, der=False) -> np.ndarray:
    """
    Sigmoid activation function

    Parameters:
        data (np.ndarray): The data to apply the function to.
        der (bool): If True, the derivative of the function is returned.
    Returns:
        np.ndarray: The data after applying the function.
    """
    if (der==True):
        f = 1 / (1 + np.exp(- data)) * (1 - 1 / (1 + np.exp(- data)))
    else:
        f = 1 / (1 + np.exp(- data))
    return f

def relu_data(data: np.ndarray, der=False) -> np.ndarray:
    """
    ReLU activation function

    Parameters:
        data (np.ndarray): The data to apply the function to.
        der (bool): If True, the derivative of the function is returned.
    Returns:
        np.ndarray: The data after applying the function.
    """
    if (der==True):
        f = np.heaviside(data, 1)
    else:
        f = np.maximum(data, 0)
    return f

def loss_data(y_pred: np.ndarray, y_truth: np.ndarray) -> float:
    """
    Computes the mean squared error loss between the true and predicted values.

    Parameters:
        y_pred (np.ndarray): The predicted values.
        y_truth (np.ndarray): The true values.
    Returns:
        float: The loss value.
    """
    return ((y_pred - y_truth) ** 2).item()

def accurancy_data(y_pred: np.ndarray, y_truth: np.ndarray) -> float:
    """
    Computes the accuracy of the model.

    Parameters:
        y_pred (np.ndarray): The predicted values.
        y_truth (np.ndarray): The true values.
    Returns:
        float: The accuracy value.
    """
    y_pred_label = 1 if y_pred >= 0.5 else 0
    return 1 if y_pred_label == y_truth else 0

def bce_data(y_true, y_pred):
    """
    Compute the binary cross-entropy error.
    
    Parameters:
        y_true (np.ndarray): True labels (binary, 0 or 1).
        y_pred (np.ndarray): Predicted probabilities for the positive class.
    
    Returns:
        float: The binary cross-entropy error.
    """
    # Clip predictions to avoid log(0) errors, using a small epsilon value.
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate the binary cross-entropy
    error = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return error