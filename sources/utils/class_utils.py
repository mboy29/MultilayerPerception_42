import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import sources.utils as utils


class Dataset:
    """
    Class to represent a dataset, with methods to process and manipulate data.
    """

    def __init__(self, name: str, raw_data: np.ndarray):
        """
        Initializes the dataset with a name and file path.

        Parameters:
            name (str): The name of the dataset.
            raw_data (np.ndarray): The raw data to process.
        Returns: None
        """
        
        # Read and store raw data
        self.name = name
        self.raw_data = raw_data
        self.ids = None
        self.labels = None
        self.features = None

        # Initialize attributes for processed features
        self.x = 0
        self.y = 0

        # Process features on initialization
        self._process_raw_data()
        self._standardize_features()
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

    def _standardize_features(self):
        """
        Private method to standardize features.

        Parameters: None
        Returns: None
        """

        self.features = utils.standardize_data(self.features)

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
    def __init__(self, 
                 data_path: str = None, 
                 layers: list = utils.DEFAULT_LAYERS, 
                 eta: float = utils.DEFAULT_LEARNING_RATE,
                 epochs: int = utils.DEFAULT_EPOCHS, 
                 batch_size: int = utils.DEFAULT_BATCH_SIZE, 
                 regulation: float = utils.DEFAULT_REGULATION, 
                 early_stopping: bool = utils.DEFAULT_EARLY_STOPPING, 
                 patience: int = utils.DEFAULT_PATIENCE):
        """
        Initializes the Multi-Layer Perceptron (MLP) model with hyperparameters.

        Parameters:
            data_path (str): The path to the dataset file, default to None.
            layers (list): The number of units in each hidden layer, default to utils.DEFAULT_LAYERS.
            eta (float): The learning rate, default to utils.DEFAULT_LEARNING_RATE.
            epochs (int): The number of training epochs, default to utils.DEFAULT_EPOCHS.
            batch_size (int): The number of samples per mini-batch, default to utils.DEFAULT_BATCH_SIZE.
            regulation (float): The L2 regularization parameter, default to utils.DEFAULT_REGULATION.
            early_stopping (bool): Enable early stopping, default to utils.DEFAULT_EARLY_STOPPING.
            patience (int): The number of epochs with no improvement to trigger early stopping, default to utils.DEFAULT_PATIENCE.
        Returns: None
        """
        # Initialize attributes
        self.train_data, self.val_data = self._split_data(data_path) if data_path else (None, None)
        self.layers = layers
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.regulation = regulation
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Early stopping variables
        self.early_stopping_epochs = None
        self.best_weights = None
        self.no_improvement_epochs = 0

        # Initialize weights and biases (1st hidden layer, 2nd hidden layer, output layer)
        self.weightH1, self.bias1 = None, None
        self.weightH2, self.bias2 = None, None
        self.weightOut, self.biasOut = None, None

        # Initialize metrics
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        self.predictions = None

    def __str__(self):
        """
        Print summary of the model.

        Parameters: None
        Returns:
            str: The summary of the model
        """
        string: str = f"{utils.BOLD_INFO}MP train: {utils.INFO}Model metrics:\n" \
                f"  - Hidden Layers: {self.layers}\n" \
                f"  - Learning Rate: {self.eta}\n" \
                f"  - Epochs: {self.epochs}\n" \
                f"  - Batch Size: {self.batch_size}\n" \
                f"  - L2 Regularization: {self.regulation}\n" \
                f"  - Early Stopping: {self.early_stopping}{utils.NC}"
        if self.early_stopping:
            string += f"{utils.INFO}\n  - Patience: {self.patience}{utils.NC}"
        return string

    def _split_data(self, file_path: Dataset, ratio: float = utils.DEFAULT_RATIO):
        """
        Private method to split the data into training and validation sets.

        Parameters:
            file_path (str): The path to the dataset file.
            ratio (float): The ratio of training data, default to utils.DEFAULT_RATIO.
        Returns:
            Dataset: The training dataset.
            Dataset: The validation dataset.
        """
        # Read the data from the file
        data = utils.read_csv(file_path)
        
        # Split the data into Malignant and Benign classes
        data_m = data[data[:, 1] == 'M']
        data_b = data[data[:, 1] == 'B']

        # Calculate the number of samples for each class based on the ratio
        train_size_m = int(len(data_m) * ratio)
        train_size_b = int(len(data_b) * ratio)

        # Split the data into training and validation sets
        data_m_train = data_m[:train_size_m]
        data_m_val = data_m[train_size_m:] if len(data_m) > train_size_m else data_m_train

        # Split the data into training and validation sets
        data_b_train = data_b[:train_size_b]
        data_b_val = data_b[train_size_b:] if len(data_b) > train_size_b else data_b_train

        # Combine the training and validation sets
        train_data = np.concatenate((data_m_train, data_b_train))
        val_data = np.concatenate((data_m_val, data_b_val))

        # Shuffle the data and return the datasets
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)
        return Dataset("train", train_data), Dataset("validation", val_data)
    

    def _init_weights(self):
        """
        Private method to initialize the weights and biases for the model.

        Parameters: None
        Returns:
            np.ndarray: The weights for the 1st hidden layer.
            np.ndarray: The biases for the 1st hidden layer.
            np.ndarray: The weights for the 2nd hidden layer.
            np.ndarray: The biases for the 2nd hidden layer.
            np.ndarray: The weights for the output layer.
            np.ndarray: The biases for the output layer.
        """

        # Initialize 1st hidden layer weights and biases
        weightH1 = np.random.randn(self.layers[0], self.train_data.features.shape[1]) * np.sqrt(2 / self.train_data.features.shape[1])
        bias1 = np.random.rand(self.layers[0])

        # Initialize 2nd hidden layer weights and biases
        weightH2 = np.random.randn(self.layers[1], self.layers[0]) * np.sqrt(2 / self.layers[0])
        bias2 = np.random.rand(self.layers[1])

        # Initialize output layer weights and biases (2 units for softmax)
        weightOut = np.random.randn(self.layers[1], 2) * np.sqrt(2 / self.layers[0])
        biasOut = np.random.rand(2)  # 2 output units

        return weightH1, bias1, weightH2, bias2, weightOut, biasOut

    
    def _update_weights(self, x, z1, z2, delta_1, delta_2, delta_Out):
        """
        Private method to update the weights and biases using backpropagation.

        Parameters:
            x (np.ndarray): The input features.
            z1 (np.ndarray): The output of the 1st hidden layer.
            z2 (np.ndarray): The output of the 2nd hidden layer.
            delta_1 (np.ndarray): The gradient of the error for the 1st hidden layer.
            delta_2 (np.ndarray): The gradient of the error for the 2nd hidden layer.
            delta_Out (np.ndarray): The gradient of the error for the output layer.
        Returns: None
        """
        # Clip gradients to prevent exploding gradients
        delta_1 = np.clip(delta_1, -10, 10)
        delta_2 = np.clip(delta_2, -10, 10)
        delta_Out = np.clip(delta_Out, -10, 10)

        # Update weights with L2 regularization
        self.weightH1 -= self.eta * (np.outer(delta_1, x) + self.regulation * self.weightH1)
        self.bias1 -= self.eta * delta_1
        self.weightH2 -= self.eta * (np.outer(delta_2, z1) + self.regulation * self.weightH2)
        self.bias2 -= self.eta * delta_2
        self.weightOut -= self.eta * (np.dot(z2[:, np.newaxis], delta_Out[:, np.newaxis].T) + self.regulation * self.weightOut)
        self.biasOut -= self.eta * delta_Out

    def _forward_propagation(self, x):
        """
        Private method to perform forward propagation on the model.

        Parameters:
            x (np.ndarray): The input features.
        Returns:
            np.ndarray: The output of the 1st hidden layer.
            np.ndarray: The output of the 2nd hidden layer.
            np.ndarray: The predicted probabilities.
        """
        z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)
        z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)
        y_pred = utils.softmax_data(np.dot(self.weightOut.T, z2) + self.biasOut)
        return z1, z2, y_pred

    def _backward_propagation(self, delta_Out, z2, z1):
        """
        Private method to perform backward propagation on the model.

        Parameters:
            delta_Out (np.ndarray): The gradient of the error for the output layer.
            z2 (np.ndarray): The output of the 2nd hidden layer.
            z1 (np.ndarray): The output of the 1st hidden layer.
        Returns:
            np.ndarray: The gradient of the error for the 1st hidden layer.
            np.ndarray: The gradient of the error for the 2nd hidden layer.
        """
        # Compute delta_2: the gradient of the error for the second hidden layer
        # delta_Out has shape (2,) and self.weightOut has shape (2, 24), so delta_2 will have shape (24,)
        delta_2 = np.dot(self.weightOut, delta_Out) * utils.relu_data(z2, der=True)  # Transpose self.weightOut for correct dimensions
        
        # Compute delta_1: the gradient of the error for the first hidden layer
        delta_1 = np.dot(self.weightH2.T, delta_2) * utils.relu_data(z1, der=True)
        
        return delta_1, delta_2

    def _train_batch(self, X_batch, Y_batch):
        """
        Private method to train the model on a mini-batch of data.

        Parameters:
            X_batch (np.ndarray): The input features for the mini-batch.
            Y_batch (np.ndarray): The labels for the mini-batch.
        Returns:
            float: The loss for the mini-batch.
            int: The number of correct predictions.
            int: The total number of predictions.
        """
        batch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Loop through the mini-batch
        for j in range(len(X_batch)):
            x = X_batch[j]
            y_true = Y_batch[j]

            # Forward propagation
            z1, z2, y_pred = self._forward_propagation(x)

            # Compute predictions and accuracy
            correct_predictions += utils.accurancy_data(y_pred, y_true)
            total_predictions += 1

            # Compute error (cross-entropy for softmax)
            delta_Out = y_pred - utils.one_hot_encode_data(y_true, 2)

            # Backpropagation
            delta_1, delta_2 = self._backward_propagation(delta_Out, z2, z1)

            # Update weights and biases
            self._update_weights(x, z1, z2, delta_1, delta_2, delta_Out)

            # Calculate the binary cross-entropy loss
            batch_loss += utils.binary_cross_entropy_data(utils.one_hot_encode_data(y_true, 2), y_pred)

        # Add L2 regularization loss (sum of squared weights)
        l2_loss = self.regulation * (np.sum(self.weightH1 ** 2) + np.sum(self.weightH2 ** 2) + np.sum(self.weightOut ** 2))
        batch_loss += l2_loss

        return batch_loss, correct_predictions, total_predictions

    def _train_epoch(self):
        """
        Private method to train the model on a full epoch of data.

        Parameters: None
        Returns:
            float: The loss for the epoch.
            float: The accuracy for the epoch.
        """
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Loop through mini-batches in training data
        for i in range(0, self.train_data.features.shape[0], self.batch_size):
            X_batch = self.train_data.features[i:i + self.batch_size]
            Y_batch = self.train_data.labels[i:i + self.batch_size]

            # Train on the current batch
            batch_loss, batch_correct_predictions, batch_total_predictions = self._train_batch(X_batch, Y_batch)

            # Accumulate loss and accuracy for the epoch
            epoch_loss += batch_loss
            correct_predictions += batch_correct_predictions
            total_predictions += batch_total_predictions

        # Average the loss over all mini-batches in the epoch
        train_accuracy = correct_predictions / total_predictions
        return epoch_loss / (self.train_data.features.shape[0] // self.batch_size), train_accuracy

    def _validate_epoch(self):
        """
        Private method to validate the model on the validation data.

        Parameters: None
        Returns:
            float: The loss for the validation set.
            float: The accuracy for the validation set.
        """
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0

        # Validation loop (no weight updates here)
        for i in range(self.val_data.features.shape[0]):
            x = self.val_data.features[i]
            y_true = self.val_data.labels[i]

            # Forward propagation on validation set
            z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)
            z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)
            y_pred = utils.softmax_data(np.dot(self.weightOut.T, z2) + self.biasOut)

            # Compute loss and accuracy for the validation set
            val_loss += utils.binary_cross_entropy_data(utils.one_hot_encode_data(y_true, 2), y_pred)
            val_correct_predictions += utils.accurancy_data(y_pred, y_true)
            val_total_predictions += 1

        # Average loss and accuracy over all validation samples
        val_accuracy = val_correct_predictions / val_total_predictions
        return val_loss / self.val_data.features.shape[0], val_accuracy
    
    def _early_stopping(self, val_loss, lowest_val_loss, epoch):
        """
        Private method to implement early stopping.

        Parameters: None
        Returns:
            int: The number of consecutive epochs with no improvement.
            tuple: The best weights and biases.
        """
        if val_loss < lowest_val_loss['value']:
            lowest_val_loss['value'] = val_loss
            lowest_val_loss['epoch'] = epoch + 1
            self.best_weights = {'epoch': epoch + 1,
                            'weights': (self.weightH1.copy(), self.bias1.copy(), 
                            self.weightH2.copy(), self.bias2.copy(), 
                            self.weightOut.copy(), self.biasOut.copy())}
            self.no_improvement_epochs = 0  # Reset counter if improvement
        else:
            self.no_improvement_epochs += 1
        if self.early_stopping and self.no_improvement_epochs >= self.patience:
            print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Early stopping triggered after { self.patience} epochs of no improvement in validation loss.{utils.NC}")
            self.early_stopping_epochs = epoch + 1
            return True
        return False
        

    def train(self):
        """
        Method to train the model on the training data and validate on the validation data.
        Implements early stopping if self.early_stopping is True.

        Parameters:
            patience (int): The number of consecutive epochs with no improvement on validation loss to stop training.
        Returns: None
        """
        print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Training model with{'out' if not self.early_stopping else ''} early stopping:{utils.NC}")

        # Initialize weights and biases
        self.weightH1, self.bias1, self.weightH2, self.bias2, self.weightOut, self.biasOut = self._init_weights()

        highest_val_accuracy = {'value': 0, 'epoch': 0}
        lowest_val_loss = {'value': float('inf'), 'epoch': 0}
        highest_train_accuracy = {'value': 0, 'epoch': 0}
        lowest_train_loss = {'value': float('inf'), 'epoch': 0}

        # Early stopping variables
        self.best_weights = None
        self.no_improvement_epochs = 0

        # Training over epochs
        for epoch in range(self.epochs):
            # 1. Train the model on the training data
            epoch_loss, train_accuracy = self._train_epoch()

            # 2. Calculate validation loss and accuracy (without updating weights)
            val_loss, val_accuracy = self._validate_epoch()

            # Track training accuracy and loss
            if train_accuracy > highest_train_accuracy['value']:
                highest_train_accuracy['value'] = train_accuracy
                highest_train_accuracy['epoch'] = epoch + 1
            
            if epoch_loss < lowest_train_loss['value']:
                lowest_train_loss['value'] = epoch_loss
                lowest_train_loss['epoch'] = epoch + 1

            # Track validation accuracy
            if val_accuracy > highest_val_accuracy['value']:
                highest_val_accuracy['value'] = val_accuracy
                highest_val_accuracy['epoch'] = epoch + 1

            # 4. Store metrics for plotting
            self.train_loss_history.append(epoch_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_accuracy)

            # 5. Print metrics for the current epoch
            self.train_metrics(epoch, epoch_loss, train_accuracy, val_loss, val_accuracy)

            # Early stopping check
            if self._early_stopping(val_loss, lowest_val_loss, epoch):
                break
        
        # Set the epoch limit for early stopping
        if self.early_stopping and self.early_stopping_epochs == None:
            self.early_stopping_epochs = self.epochs

        # Restore the best weights
        if self.early_stopping:
            if self.best_weights is not None:
                self.weightH1, self.bias1, self.weightH2, self.bias2, self.weightOut, self.biasOut = self.best_weights['weights']
                print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Restored best weights from epoch {self.best_weights['epoch']}.{utils.NC}")
            else:
                print(f"{utils.BOLD_INFO}MP train: {utils.INFO}No improvement in validation loss, best weights not restored.{utils.NC}")

        print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Training complete.{utils.NC}")

        # Print final metrics
        self.train_final_metrics(highest_val_accuracy, lowest_val_loss, highest_train_accuracy, lowest_train_loss)


    def train_plot(self, plot_path: str = utils.DEFAULT_PLOT_PATH):
        """
        Method to plot the training and validation metrics, this includes loss and accuracy.

        Parameters:
            plot_path (str): The path to save the plot, default to utils.DEFAULT_PLOT_PATH.
        Returns: None
        """

        # Check if early stopping was triggered to set the epoch limit
        if self.early_stopping: epoch = self.early_stopping_epochs
        else: epoch = self.epochs

        # Plot metrics
        plt.figure(figsize=(12, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 1), self.train_loss_history, 'b', label='Train Loss')
        plt.plot(range(1, epoch + 1), self.val_loss_history, 'r', label='Validation Loss')
        plt.title(f'Train vs Validation Loss {"(with early stopping)" if self.early_stopping else ""}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 1), self.train_accuracy_history, 'b', label='Train Accuracy')
        plt.plot(range(1, epoch + 1), self.val_accuracy_history, 'r', label='Validation Accuracy')
        plt.title(f'Train vs Validation Accuracy {"(with early stopping)" if self.early_stopping else ""}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        plt.tight_layout()

        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
        plt.savefig(plot_path)
        print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Plot saved to {plot_path}{utils.NC}")

    def train_metrics(self, epoch: int, epoch_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float):
        """
        Method to print the training metrics for the current epoch.

        Parameters:
            epoch (int): The current epoch.
            epoch_loss (float): The loss for the current epoch.
            train_accuracy (float): The accuracy for the training data.
            val_loss (float): The loss for the validation data.
            val_accuracy (float): The accuracy for the validation data.
        Returns: None
        """
        print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Epoch {epoch+1}/{self.epochs} "
                  f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}{utils.NC}")

    def train_final_metrics(self, highest_val_accuracy, lowest_val_loss, highest_train_accuracy, lowest_train_loss):
        """
        Method to print the final training metrics.

        Parameters:
            highest_val_accuracy (dict): The highest validation accuracy and epoch.
            lowest_val_loss (dict): The lowest validation loss and epoch.
            highest_train_accuracy (dict): The highest training accuracy and epoch.
            lowest_train_loss (dict): The lowest training loss and epoch.
        Returns: None
        """
        print(f"\n{self}")
        print(f"\n{utils.BOLD_INFO}MP train: {utils.INFO}Training metrics:")
        print(f"  - Highest Validation Accuracy {highest_val_accuracy['value']:.4f} at Epoch {highest_val_accuracy['epoch']}")
        print(f"  - Lowest Validation Loss {lowest_val_loss['value']:.6f} at Epoch {lowest_val_loss['epoch']}")
        print(f"  - tHighest Training Accuracy {highest_train_accuracy['value']:.4f} at Epoch {highest_train_accuracy['epoch']}")
        print(f"  - Lowest Training Loss {lowest_train_loss['value']:.6f} at Epoch {lowest_train_loss['epoch']}{utils.NC}\n")

    def save_model(self, model_path: str = utils.DEFAULT_MODEL_PATH):
        """
        Method to save the model weights and biases to a file.

        Parameters:
            model_path (str): The path to save the model, default to utils.DEFAULT_MODEL_PATH.
        Returns: None
        """
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        np.savez(model_path, w1=self.weightH1, b1=self.bias1, w2=self.weightH2, b2=self.bias2, wOut=self.weightOut, bOut=self.biasOut)
        print(f"{utils.BOLD_INFO}MP train: {utils.INFO}Model saved to {model_path}{utils.NC}")
    
    def load_model(self, model_path: str = utils.DEFAULT_MODEL_PATH):
        """
        Method to load the model weights and biases from a file.
        
        Parameters:
            model_path (str): The path to load the model from, default to utils.DEFAULT_MODEL_PATH. 
        Returns: None
        """
        model_data = np.load(model_path)

        self.weightH1 = model_data['w1']
        self.bias1 = model_data['b1']
        self.weightH2 = model_data['w2']
        self.bias2 = model_data['b2']
        self.weightOut = model_data['wOut']
        self.biasOut = model_data['bOut']
        print(f"{utils.BOLD_INFO}MP predict: {utils.INFO}Model loaded from {model_path}{utils.NC}")

    def predict(self, X_test, Y_test):
        """
        Method to predict the labels for the test data.

        Parameters:
            X_test (np.ndarray): The test data to predict on.
        Returns: None
        """
        predictions = []

        # Loop in all the test samples
        for I in range(0, X_test.shape[0]):  
            
            # Input the data 
            x = X_test[I]
            
            # Feed forward
            z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)  # first hidden layer
            z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)  # second hidden layer
            y = utils.softmax_data(np.dot(self.weightOut.T, z2) + self.biasOut)
            
            # Append the prediction; class with the highest probability
            predictions.append(np.argmax(y))
        
        self.predictions = np.array(predictions)
        print(f"{utils.BOLD_INFO}MP predict: {utils.INFO}Predictions completed{utils.NC}")

        # Print metrics
        self._predict_metrics(Y_test, X_test)

    def _predict_metrics(self, Y_test, X_test):
        """
        Private method to calculate and print the accuracy and binary cross-entropy loss for the predictions.

        Parameters:
            Y_test (np.ndarray): The true labels for the test data.
            X_test (np.ndarray): The test data to predict on.
        Returns: None
        """

        cm = confusion_matrix(Y_test, self.predictions)

        # Calculate accuracy
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        # Initialize an array to store predicted probabilities
        predicted_probs = np.zeros(Y_test.shape)

        # Calculate predicted probabilities (feed-forward through the model)
        for i in range(X_test.shape[0]):
            x = X_test[i]
            z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)
            z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)
            y_pred = utils.softmax_data(np.dot(self.weightOut.T, z2) + self.biasOut)
            predicted_probs[i] = y_pred[1]

        # Calculate binary cross-entropy error
        bce_error = utils.binary_cross_entropy_data(Y_test, predicted_probs)
        
        # Print the metrics
        print(f"\n{utils.BOLD_INFO}MP predict: {utils.INFO}Predictions metrics:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Binary Cross-Entropy Loss: {bce_error:.6f}{utils.NC}\n")



    def predict_plot(self, Y_test, cm_path: str = utils.DEFAULT_CONFUSION_MATRIX_PATH):
        """
        Method to plot the confusion matrix for the predictions.

        Parameters:
            Y_test (np.ndarray): The true labels for the test data.
            cm_path (str): The path to save the confusion matrix plot, default to utils.DEFAULT_CONFUSION_MATRIX_PATH.
        Returns: None
        """
        
        # Calculate confusion matrix
        cm = confusion_matrix(Y_test, self.predictions  )

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

        if not os.path.exists(os.path.dirname(cm_path)):
            os.makedirs(os.path.dirname(cm_path))
        plt.savefig(cm_path)
        print(f"{utils.BOLD_INFO}MP predict: {utils.INFO}Confusion Matrix saved to {cm_path}{utils.NC}")

    def save_predictions(self, path: str = utils.DEFAULT_PREDICTIONS_PATH):
        """
        Method to save the predictions to a file.

        Parameters:
            path (str): The path to save the predictions, default to utils.DEFAULT_PREDICTIONS_PATH.
        Returns: None
        """
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        labelled_predictions = np.where(self.predictions == 1, 'M', 'B')
        np.savetxt(utils.DEFAULT_PREDICTIONS_PATH, labelled_predictions, delimiter=",", comments='', fmt="%s")
        print(f"{utils.BOLD_INFO}MP predict: {utils.INFO}Predictions saved to {path}{utils.NC}")
