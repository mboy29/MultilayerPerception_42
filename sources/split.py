import argparse
import numpy as np
import sources.utils as utils

def split_manually(data: np.ndarray, train_ratio : float = utils.DEFAULT_TRAIN_RATIO, val_ratio: float = utils.DEFAULT_VAL_RATIO, test_ratio : float = utils.DEFAULT_TEST_RATIO) -> tuple:
    """
    Splits the data manually into training, validation, and testing sets using the given ratios.

    Parameters:
        data (np.ndarray): The dataset to split.
        train_ratio (float): The ratio of data to be included in the training set.
        val_ratio (float): The ratio of data to be included in the validation set.
        test_ratio (float): The ratio of data to be included in the testing set.
    Returns:
        np.ndarray: Training set.
        np.ndarray: Validation set.
        np.ndarray: Testing set.
    """
    if round(train_ratio + val_ratio + test_ratio, 2) != 1:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")
    
    shuffled_indices = np.random.permutation(data.shape[0])
    
    # Split the dataset into training, validation, and testing sets
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    
    train = data[train_indices]
    val = data[val_indices]
    test = data[test_indices]

    return train, val, test

def split_data(data: np.ndarray, train_ratio : float = utils.DEFAULT_TRAIN_RATIO, val_ratio: float = utils.DEFAULT_VAL_RATIO, test_ratio : float = utils.DEFAULT_TEST_RATIO) -> tuple:
    """
    Splits the dataset into training, validation, and testing sets,
    while maintaining the ratio of 'M' and 'B' labels.

    Parameters:
        data (np.ndarray): The dataset to split.
        train_ratio (float): The ratio of the training set (default 0.8).
        val_ratio (float): The ratio of the validation set (default 0.2).
        test_ratio (float): The ratio of the test set (default 0.0).
    Returns:
        np.ndarray: The training set.
        np.ndarray: The validation set.
        np.ndarray: The testing set.
    """

    print(f"MP split: Shuffling data set")
    np.random.shuffle(data)

    print(f"MP split: Splitting data set into Malignant and Benign")
    data_m = data[data[:, 1] == 'M']
    data_b = data[data[:, 1] == 'B']
    print(f"MP split: Splitting data set into training, validation, and testing")

    # Perform the split for both malignant and benign data
    train_m, val_m, test_m = split_manually(data_m, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    train_b, val_b, test_b = split_manually(data_b, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    print(f"MP split: Combining training, validation, and testing sets")

    # Concatenate the corresponding sets
    train = np.concatenate((train_m, train_b))
    val = np.concatenate((val_m, val_b))
    test = np.concatenate((test_m, test_b))

    print(f"MP split: Shuffling training, validation, and testing sets")

    # Shuffle the resulting sets
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    return train, val, test

def split(dataset: str, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """
    Splits the dataset into training, validation, and testing sets.

    Parameters:
        dataset (str): The path to the dataset file.
        train_ratio (float): The ratio of data to be included in the training set.
        val_ratio (float): The ratio of data to be included in the validation set.
        test_ratio (float): The ratio of data to be included in the testing set.
    Returns:
        None
    """
    try:
        if round(train_ratio + val_ratio + test_ratio, 2) != 1:
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")
        
        data = utils.read_csv(dataset)
        print(f"MP split: Splitting dataset...")
        train, val, test = split_data(data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        
        # Save the datasets to the specified paths
        utils.save_csv(train, utils.DEFAULT_TRAIN_PATH)
        utils.save_csv(val, utils.DEFAULT_VALIDATION_PATH)
        utils.save_csv(test, utils.DEFAULT_TEST_PATH)
        
        print(f"MP split: Splitting complete, training, validation, and testing sets saved to '{utils.DEFAULT_DATASETS_DIR_PATH}' directory")
    except Exception as e:
        raise Exception(f"split: error: {e}")
