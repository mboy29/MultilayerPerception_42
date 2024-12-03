import argparse
import numpy as np
import os
import utils as utils

ERROR='\033[38;2;255;173;194m'
INFO='\033[38;2;204;219;253m'
SUCCESS='\033[38;2;207;225;185m'
NC='\033[0m'
DATASETS_DIR_PATH = "datasets"
TRAIN_FILE_PATH = os.path.join(DATASETS_DIR_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(DATASETS_DIR_PATH, "test.csv")

def split_manually(data, ratio=0.8):
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

def split(data, ratio=0.8):
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

    print(f"{INFO}split.py: Splitting data set into Malignant and Benign{NC}")
    data_m = data[data[:, 1] == 'M']
    data_b = data[data[:, 1] == 'B']
    print(f"{INFO}split.py: Splitting data set into training and testing{NC}")
    

    train_m, test_m = split_manually(data_m, ratio=ratio)
    train_b, test_b = split_manually(data_b, ratio=ratio)

    print(f"{INFO}split.py: Combining training and testing sets{NC}")

    train = np.concatenate((train_m, train_b))
    test = np.concatenate((test_m, test_b))

    print(f"{INFO}split.py: Shuffling training and testing sets{NC}")

    np.random.shuffle(train)
    np.random.shuffle(test)

    return train, test

def main(args):
    data = utils.read_csv(args.dataset)
    print(f"{INFO}split.py: Splitting dataset...{NC}")
    train, test = split(data)
    utils.save_csv(train, TRAIN_FILE_PATH)
    utils.save_csv(test, TEST_FILE_PATH)
    print(f"{SUCCESS}split.py: Splitting complete, training and testing sets saved to '{DATASETS_DIR_PATH}' directory{NC}")
    

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="Split",
            description="Splits a dataset into distinct training and testing subsets based on labels.",
        )
        parser.add_argument("dataset", type=str, help="Path to the CSV file containing the dataset")
        args = parser.parse_args()

        main(args)
    except Exception as e:
        print(f"{ERROR}Error: {e}{NC}")
