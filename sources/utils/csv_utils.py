import os
import numpy as np

def check_csv(file_path):
    """
    Check if the file exists and is readable.

    Parameters:
        file_path (str): The file to check.
    Returns: None
    Raises:
        Exception: If the file does not exist.
        Exception: If the file is not readable.
    """
    if not os.path.exists(file_path):
        raise Exception(f"File does not exist: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise Exception(f"File is not readable: {file_path}")
    if os.path.isdir(file_path):
        raise Exception(f"Path is a directory, not a file: {file_path}")
    if not file_path.lower().endswith('.csv'):
        raise ValueError("The file provided is not a CSV file.")


def read_csv(file_path):
    """
    Reads a CSV file using numpy and validates its structure:
    - The first column must contain integers.
    - The second column must contain 'M' or 'B'.
    - All other columns must contain floats.

    Parameters:
        file_path (str): The path to the CSV file.
    Returns:
        np.ndarray: The contents of the CSV file as a numpy array.
    Raises:
        ValueError: If the file contains invalid data or structure.
        Exception: For other unforeseen errors.
    """
    check_csv(file_path)
    try:
        # Load the file as a string array
        data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
        if data.size == 0:
            raise ValueError("CSV file is empty")
        
        if np.any(data == ''):
            raise ValueError("CSV file contains missing values")
        
        try:
            data[:, 0].astype(int)
        except ValueError:
            raise ValueError("First column must only contain integers")

        if not np.all(np.isin(data[:, 1], ['M', 'B'])):
            raise ValueError("Second column must contain only be 'M' or 'B'")
        
        try:
            data[:, 2:].astype(float)
        except ValueError:
            raise ValueError("All columns except the first two must contain only floats.")
        return data

    except Exception as e:
        raise Exception(f"{e}: {file_path}")

def save_csv(data, path):
    """
    Saves a numpy array to a CSV file.

    Parameters:
        data (np.ndarray): The data to save.
        path (str): The path to save the data to.
    Returns: None
    Raises:
        ValueError: If the data is not a numpy array.
        Exception: If the file cannot be written to.
    """
    try:
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        np.savetxt(path, data, delimiter=',', fmt='%s', header='id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst')
    except Exception as e:
        raise Exception(f"Error saving file: {e}")


