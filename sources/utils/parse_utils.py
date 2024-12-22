import sys
import ast
import argparse
import sources as utils

class CustomArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser class that provides personalized error messages and usage output.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the custom ArgumentParser with custom program name and description.
        
        Parameters:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        Returns: None
        """
        kwargs['prog'] = kwargs.get('prog', "MP")
        super().__init__(*args, **kwargs)
        self.description = "A simple multilayer perceptron for classifying breast cancer data."
        self.epilog = "Made by: mboy - 42 Nice"
        self.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode.")

    def error(self, message):
        """
        Overrides the default error method to provide custom error messages.

        Parameters:
            message (str): The error message to display.
        Returns: None
        """
        sys.stderr.write(f"{utils.ERROR}MP error: {message}{utils.NC}\n")
        self.print_usage()
        exit(2)

    def print_usage(self):
        """
        Overrides the usage message with custom formatting.

        Parameters: None
        Returns: None
        """
        usage = self.format_usage()[7:].capitalize()
        sys.stdout.write(f"{utils.BOLD_INFO}MP usage: {utils.INFO}{usage}{utils.NC}")

    def print_help(self):
        """
        Overrides the help message to include custom headers and colors.

        Parameters:
            file: The file to write the help message to.
        Returns: None
        """
        help = self.format_help()
        print(f"{utils.BOLD_INFO}MP help: {utils.INFO}{help}{utils.NC}")

def list_parse(value: str) -> list:
    """
    Helper function to parse the layers argument into a list of integers.

    Parameters:
        value (str): The string value to parse.
    Returns:
        list: The parsed list of integers
    """
    try:
        layers = ast.literal_eval(value)
        if isinstance(layers, list) and all(isinstance(x, int) for x in layers):
            if len(layers) != 2:
                raise ValueError("The number of hidden layers must be 2.")
            return layers
        raise ValueError("The layers must be a list of integers.")
    except Exception as e:
        raise ValueError(f"Invalid input for layers: {e}")

def unint_parse(value: str) -> int:
    """
    Helper function to parse unsigned integer arguments.

    Parameters:
        value (str): The string value to parse.
    Returns:
        int: The parsed unsigned integer
    """
    try:
        num = int(value)
        if num >= 0:
            return num
        raise ValueError("The number must be an unsigned integer.")
    except Exception as e:
        raise ValueError(f"Invalid input for unsigned integer: {e}")

def unfloat_parse(value: str) -> float:
    """
    Helper function to parse unsigned float arguments.

    Parameters:
        value (str): The string value to parse.
    Returns:
        float: The parsed unsigned float
    """
    try:
        num = float(value)
        if num >= 0:
            return num
        raise ValueError("The number must be an unsigned float.")
    except Exception as e:
        raise ValueError(f"Invalid input for unsigned float: {e}")

def parse() -> argparse.Namespace:
    """
    Parses command-line arguments and returns them as a namespace.

    Parameters: None
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = CustomArgumentParser()

    # Sub-command parsers
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands")

    # Sub-parser for the "split" command
    split_parser = subparsers.add_parser("split", help="Split the dataset into training and testing sets")
    split_parser.add_argument("-p", "--data_path", type=str, required=False, help=f"Path to the CSV file containing the dataset (default: '{utils.DEFAULT_DATASET_PATH}')", default=utils.DEFAULT_DATASET_PATH)
    split_parser.add_argument("-r", "--ratio", type=float, required=False, help="The ratio of data to be included in the training set", default=utils.DEFAULT_RATIO)

    # Sub-parser for the "train" command
    train_parser = subparsers.add_parser("train", help="Train the multilayer perceptron model")
    train_parser.add_argument("-p", "--data_path", type=str, required=False, help=f"Path to the CSV file containing the training set (default: '{utils.DEFAULT_TRAIN_PATH}')", default=utils.DEFAULT_TRAIN_PATH)
    train_parser.add_argument("-l", "--layers_size", type=list_parse, required=False,   help="The size of the 2 hidden layers", default=utils.DEFAULT_LAYERS)
    train_parser.add_argument("-eta", "--learning_rate", type=unfloat_parse, required=False, help="The learning rate of the model", default=utils.DEFAULT_LEARNING_RATE)
    train_parser.add_argument("-e", "--epochs", type=unint_parse, required=False, help="The number of epochs to train the model", default=utils.DEFAULT_EPOCHS)
    train_parser.add_argument("-b", "--batch_size", type=unint_parse, required=False, help="The batch size to train the model", default=utils.DEFAULT_BATCH_SIZE)
    train_parser.add_argument("-d", "--decay", type=unfloat_parse, required=False, help="The decay of the model", default=utils.DEFAULT_DECAY)

    # Sub-parser for the "predict" command
    predict_parser = subparsers.add_parser("predict", help="Predict the labels of the testing set using the trained model")
    predict_parser.add_argument("-p", "--data_path", type=str, required=False, help=f"Path to the CSV file containing the testing set (default: '{utils.DEFAULT_TEST_PATH}')", default=utils.DEFAULT_TEST_PATH)

    return parser.parse_args()
