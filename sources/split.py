import argparse
import numpy as np
import sources.utils as utils

def split_manually(data, ratio=0.8) -> tuple:
    """
    Splits the data manually into training and testing sets using the training ratio.

    Parameters:
        data (np.ndarray): The dataset to split.
        ratio (float): The ratio of data to be included in the training set.
    Returns:
        np.ndarray: Training set.
        np.ndarray: Testing set.
    """
    shuffled_indices = np.random.permutation(data.shape[0])
    train_size = int(len(data) * ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train = data[train_indices]
    test = data[test_indices]

    return train, test

def split_data(data, ratio=0.8) -> tuple:
    """
    Splits the dataset into training and testing, 
    while maintaining the ratio of 'M' and 'B' labels.

    Parameters:
        data (np.ndarray): The dataset to split.
        ratio (float): The ratio of the training set (default 0.8).
    Returns:
        np.ndarray: The training set.
        np.ndarray: The testing set.
    """

    print(f"MP split: Shuffling data set")
    np.random.shuffle(data)

    print(f"MP split: Splitting data set into Malignant and Benign")
    data_m = data[data[:, 1] == 'M']
    data_b = data[data[:, 1] == 'B']
    print(f"MP split: Splitting data set into training and testing")
    

    train_m, test_m = split_manually(data_m, ratio=ratio)
    train_b, test_b = split_manually(data_b, ratio=ratio)

    print(f"MP split: Combining training and testing sets")

    train = np.concatenate((train_m, train_b))
    test = np.concatenate((test_m, test_b))

    print(f"MP split: Shuffling training and testing sets")

    np.random.shuffle(train)
    np.random.shuffle(test)

    return train, test

def split(dataset: str, ratio: float) -> None:
    """
    Splits the dataset into training and testing sets.

    Parameters:
        dataset (str): The path to the dataset file.
    Returns:
        None
    """

    try:
        data = utils.read_csv(dataset)
        print(f"MP split: Splitting dataset...")
        train, test = split_data(data, ratio=ratio)
        utils.save_csv(train, utils.DEFAULT_TRAIN_PATH)
        utils.save_csv(test, utils.DEFAULT_VALIDATION_PATH)
        print(f"MP split: Splitting complete, training and testing sets saved to '{utils.DEFAULT_DATASETS_DIR_PATH}' directory")
    except Exception as e:
        raise Exception(f"split: error: {e}")
