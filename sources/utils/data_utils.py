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
