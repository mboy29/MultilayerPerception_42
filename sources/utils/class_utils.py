import numpy as np
import sources.utils as utils

class Dataset:
    """
    Class to represent a dataset, with methods to process and manipulate data.
    """

    def __init__(self, name: str, file_path: str):
        """
        Initializes the dataset with a name and file path.

        Parameters:
            name (str): The name of the dataset.
            file_path (str): The path to the dataset file.
        Returns: None
        """
        
        # Read and store raw data
        self.name = name
        self.raw_data = utils.read_csv(file_path)
        self.ids = None
        self.labels = None
        self.features = None

        # Initialize attributes for processed features
        self.x = 0
        self.y = 0

        # Process features on initialization
        self._process_raw_data()
        self._normalize_features()
        self._binarize_labels()

    def _process_raw_data(self):
        """
        Private method to extract ids, labels, and features from raw input.

        Parameters: None
        Returns: None
        """
        
        # Shuffle data
        np.random.shuffle(self.raw_data)

        # Transpose .T every row except the first and second one
        
        # Extract ids, labels, and features
        self.ids = self.raw_data[:, 0].astype(int)
        self.labels = self.raw_data[:, 1].astype(str)
        self.features = self.raw_data[:, 2:].astype(float) # Might not work

        # Store dimensions
        self.x, self.y = self.features.shape

    def _normalize_features(self):
        """
        Private method to normalize features.

        Parameters: None
        Returns: None
        """

        self.features = utils.normalize_data(self.features)

    def _binarize_labels(self):
        """
        Private method to convert labels to binary (M=1, B=0).

        Parameters: None
        Returns: None
        """

        self.labels = np.where(self.labels == 'M', 1, 0)

    def __str__(self, n=5):
        """
        Print summary of the dataset.

        Parameters:
            n (int): The number of samples to display (default 5).
        Returns:
            str: The summary of the dataset
        """

        return f"Dataset '{self.name}' Summary:\n" \
                f" - Number of samples (x): {self.x}\n" \
                f" - Number of features (y): {self.y}"

class MLP:
    def __init__(self, input_layer: int, 
                 hidden_layers: list = [24, 24],  # Set 2 hidden layers by default
                 output_layer: int = 1,            # Binary classification (output = 1)
                 iterations: int = 1000, 
                 learning_rate: float = 0.01):
        """
        Initializes the MLP with hyperparameters.

        Parameters:
            input_layer (int): The number of features in the input layer.
            hidden_layers (list): List with the number of neurons in hidden layers.
            output_layer (int): Number of output neurons (1 for binary classification).
            iterations (int): Number of training iterations.
            learning_rate (float): Learning rate for gradient descent.
        """
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.layers = [input_layer] + hidden_layers + [output_layer]

        self.iterations = iterations
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights, self.biases = self._init_weights()
