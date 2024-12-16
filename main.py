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

    train_parser = subparsers.add_parser("train", help="Train the multilayer perceptron model")
    predict_parser = subparsers.add_parser("predict", help="Predict the labels of the testing set using the trained model")

    return parser.parse_args()


def main():
    args = parse()
    if args.command == "split":
        utils.split(args.dataset_path)
    elif args.command == "train":
        utils.train()
    elif args.command == "test":
        utils.predict()
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"MP {e}")
