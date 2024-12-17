import argparse
import sources as utils

def parse() -> argparse.Namespace:
    """
    Parses the command line arguments, and returns the parsed arguments.

    Parameters: None
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    
    parser = argparse.ArgumentParser(
        prog="MP",
        epilog="Made by: ndesprez x mboy - 42 Nice",
        description="A simple multilayer perceptron for classifying breast cancer data",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands")

    split_parser = subparsers.add_parser("split", help="Split the dataset into training and testing sets")
    split_parser.add_argument("--dataset_path", type=str, required=False, help=f"Path to the CSV file containing the dataset, default to '{utils.DEFAULT_DATASET_PATH}'", default=utils.DEFAULT_DATASET_PATH)
    split_parser.add_argument("--train_ratio", type=float, required=False, help="Training ratio of the training set, default to 0.8", default=utils.DEFAULT_TRAIN_RATIO)
    split_parser.add_argument("--val_ratio", type=float, required=False, help="Validation ratio of the training set, default to 0.1", default=utils.DEFAULT_VAL_RATIO)
    split_parser.add_argument("--test_ratio", type=float, required=False, help="Testing ratio of the training set, default to 0.1", default=utils.DEFAULT_TEST_RATIO)

    train_parser = subparsers.add_parser("train", help="Train the multilayer perceptron model")
    train_parser.add_argument("--train_path", type=str, required=False, help=f"Path to the CSV file containing the training set, default to '{utils.DEFAULT_TRAIN_PATH}'", default=utils.DEFAULT_TRAIN_PATH)
    train_parser.add_argument("--validation_path", type=str, required=False, help=f"Path to the CSV file containing the validation set, default to '{utils.DEFAULT_VALIDATION_PATH}'", default=utils.DEFAULT_VALIDATION_PATH)

    predict_parser = subparsers.add_parser("predict", help="Predict the labels of the testing set using the trained model")
    predict_parser.add_argument("--test_path", type=str, required=False, help=f"Path to the CSV file containing the testing set, default to '{utils.DEFAULT_TEST_PATH}'", default=utils.DEFAULT_TEST_PATH)

    return parser.parse_args()


def main():
    args = parse()
    if args.command == "split":
        utils.split(args.dataset_path, args.train_ratio, args.val_ratio, args.test_ratio)
    elif args.command == "train":
        utils.train(args.train_path, args.validation_path)
    elif args.command == "predict":
        utils.predict(args.test_path)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"MP {e}")
