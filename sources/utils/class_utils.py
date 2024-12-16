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
        self.results = None
        self.data = None

        # Initialize attributes for processed data
        self.x = 0
        self.y = 0

        # Process data on initialization
        self._process_raw_data()
        self._normalize_data()
        self._binarize_results()

    def _process_raw_data(self):
        """
        Private method to extract ids, results, and data from raw input.

        Parameters: None
        Returns: None
        """
        
        # Shuffle data
        np.random.shuffle(self.raw_data)

        # Extract ids, results, and features
        self.ids = self.raw_data[:, 0].astype(int)
        self.results = self.raw_data[:, 1].astype(str)
        self.data = self.raw_data[:, 2:].astype(float)

        # Store dimensions
        self.x, self.y = self.data.shape

    def _normalize_data(self):
        """
        Private method to normalize feature data.

        Parameters: None
        Returns: None
        """

        self.data = utils.normalize_data(self.data)

    def _binarize_results(self):
        """
        Private method to convert results to binary (M=1, B=0).

        Parameters: None
        Returns: None
        """

        self.results = np.where(self.results == 'M', 1, 0)

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