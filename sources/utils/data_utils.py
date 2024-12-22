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

def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardizes the data using z-score normalization, applied column-wise.

    Parameters:
        data (np.ndarray): The data to standardize.
    Returns:
        np.ndarray: The standardized data.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

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
    return np.mean((y_pred - y_truth) ** 2)

def accurancy_data(y_pred: np.ndarray, y_truth: np.ndarray) -> float:
    """
    Computes the accuracy of the model.

    Parameters:
        y_pred (np.ndarray): The predicted values.
        y_truth (np.ndarray): The true values.
    Returns:
        float: The accuracy value.
    """
    predicted_class = np.argmax(y_pred)
    return np.sum(predicted_class == y_truth)

def binary_cross_entropy_data(y_true, y_pred):
    """
    Binary Cross Entropy loss function.
    
    Parameters:
        y_true (numpy.ndarray): The true labels.
        y_pred (numpy.ndarray): The predicted labels (probabilities between 0 and 1).
    Returns:
        float: The loss value.
    """
    # Debug prints, useful during development, can be commented out in production
    # print("true", y_true.shape, y_true)
    # print("pred", y_pred.shape, y_pred)
    
    # Numerical stability: Clip y_pred to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute Binary Cross Entropy
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss

def one_hot_encode_data(y, num_classes=2):
    """
    One-hot encodes the target labels.

    Parameters:
        y (numpy.ndarray): The target labels.
        num_classes (int): The number of classes.
    Returns:
        numpy.ndarray: The one-hot encoded labels.
    """
    return np.eye(num_classes)[y]

def softmax_data(z):
    """
    Softmax activation function for the output layer.

    Parameters:
        z (numpy.ndarray): The input vector for the output layer.
    Returns:
        numpy.ndarray: The probabilities for each class.
    """
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)