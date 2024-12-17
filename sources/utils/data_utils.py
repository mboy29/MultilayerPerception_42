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

