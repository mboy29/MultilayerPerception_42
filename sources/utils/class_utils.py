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
    def __init__(self, train_data: Dataset = None, val_data: Dataset = None, layers: list = [24, 24], eta: float = 0.0314, epochs: int = 84, batch_size: int = 8):
        
        # Initialize attributes
        self.train_data = train_data
        self.val_data = val_data
        self.layers = layers
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size

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

    def _init_weights(self):

        # Initialize 1st hidden layer weights and biases
        weightH1 = 2 * np.random.rand(self.layers[0], self.train_data.features.shape[1]) - 0.5
        bias1 = np.random.rand(self.layers[0])

        # Initialize 2nd hidden layer weights and biases
        weightH2 = 2 * np.random.rand(self.layers[1], self.layers[0]) - 0.5
        bias2 = np.random.rand(self.layers[1])

        # Initialize output layer weights and biases
        weightOut = 2 * np.random.rand(self.layers[1]) - 0.5
        biasOut = np.random.rand(1)

        return weightH1, bias1, weightH2, bias2, weightOut, biasOut

    def _update_weights(self, x, z1, z2, delta_1, delta_2, delta_Out):
        
        # Update 1st hidden layer weights and biases
        self.weightH1 -= self.eta * np.outer(delta_1, x)
        self.bias1 -= self.eta * delta_1
        
        # Update 2nd hidden layer weights and biases
        self.weightH2 -= self.eta * np.outer(delta_2, z1)
        self.bias2 -= self.eta * delta_2
        
        # Update output layer weights and biases
        self.weightOut -= self.eta * delta_Out * z2
        self.biasOut -= self.eta * delta_Out


    def _forward_propagation(self, x):
        z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)
        z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)
        y_pred = utils.sigmoid_data(np.dot(self.weightOut, z2) + self.biasOut)
        return z1, z2, y_pred

    def _backward_propagation(self, delta_Out, z2, z1):
        delta_2 = delta_Out * self.weightOut * utils.relu_data(z2, der=True)
        delta_1 = np.dot(delta_2, self.weightH2) * utils.relu_data(z1, der=True)
        return delta_1, delta_2
    
    def _train_batch(self, X_batch, Y_batch):
        batch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for j in range(len(X_batch)):
            x = X_batch[j]
            y_true = Y_batch[j]

            # Forward propagation
            z1, z2, y_pred = self._forward_propagation(x)

            # Compute predictions and accuracy
            correct_predictions += utils.accurancy_data(y_pred, y_true)
            total_predictions += 1

            # Compute error
            delta_Out = 2 * (y_pred - y_true) * utils.sigmoid_data(y_pred, der=True)

            # Backpropagation
            delta_1, delta_2 = self._backward_propagation(delta_Out, z2, z1)

            # Update weights and biases
            self._update_weights(x, z1, z2, delta_1, delta_2, delta_Out)

            # Accumulate batch loss
            batch_loss += utils.loss_data(y_pred, y_true)

        return batch_loss, correct_predictions, total_predictions

    def _train_epoch(self):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Mini-batch gradient descent
        for i in range(0, self.train_data.features.shape[0], self.batch_size):
            X_batch = self.train_data.features[i:i + self.batch_size]
            Y_batch = self.train_data.labels[i:i + self.batch_size]

            # Train on the current batch
            batch_loss, batch_correct_predictions, batch_total_predictions = self._train_batch(X_batch, Y_batch)

            # Accumulate loss and accuracy
            epoch_loss += batch_loss / len(X_batch)
            correct_predictions += batch_correct_predictions
            total_predictions += batch_total_predictions

        train_accuracy = correct_predictions / total_predictions
        return epoch_loss, train_accuracy

    def _validate_epoch(self):
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0

        # Validation loop
        for i in range(self.val_data.features.shape[0]):
            x = self.val_data.features[i]
            y_true = self.val_data.labels[i]

            # Forward propagation on validation set
            z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)
            z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)
            y_pred = utils.sigmoid_data(np.dot(self.weightOut, z2) + self.biasOut)

            # Compute loss and accuracy
            val_loss += utils.loss_data(y_pred, y_true)
            val_correct_predictions += utils.accurancy_data(y_pred, y_true)
            val_total_predictions += 1

        val_accuracy = val_correct_predictions / val_total_predictions
        return val_loss / self.val_data.features.shape[0], val_accuracy

    def train(self):
        # Initialize weights and biases
        self.weightH1, self.bias1, self.weightH2, self.bias2, self.weightOut, self.biasOut = self._init_weights()

        highest_val_accuracy = 0
        highest_val_accuracy_epoch = 0
        lowest_val_loss = float('inf')
        lowest_val_loss_epoch = 0

        highest_train_accuracy = 0
        highest_train_accuracy_epoch = 0
        lowest_train_loss = float('inf')
        lowest_train_loss_epoch = 0

        # Training over epochs
        for epoch in range(self.epochs):
            epoch_loss, train_accuracy = self._train_epoch()

            # Validation loss and accuracy
            val_loss, val_accuracy = self._validate_epoch()

            # Store the best model
            if val_accuracy > highest_val_accuracy:
                highest_val_accuracy = val_accuracy
                highest_val_accuracy_epoch = epoch
            
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                lowest_val_loss_epoch = epoch
            
            if train_accuracy > highest_train_accuracy:
                highest_train_accuracy = train_accuracy
                highest_train_accuracy_epoch = epoch
            
            if epoch_loss < lowest_train_loss:
                lowest_train_loss = epoch_loss
                lowest_train_loss_epoch = epoch

            # Store the metrics
            self.train_loss_history.append(epoch_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_accuracy)

            # Print metrics
            self.train_metrics(epoch, epoch_loss, train_accuracy, val_loss, val_accuracy)

        print("MP train: Training completed")
        print(f"MP train: Highest Validation Accuracy {highest_val_accuracy:.4f} at Epoch {highest_val_accuracy_epoch+1}")
        print(f"MP train: Lowest Validation Loss {lowest_val_loss:.6f} at Epoch {lowest_val_loss_epoch+1}")
        print(f"MP train: Highest Train Accuracy {highest_train_accuracy:.4f} at Epoch {highest_train_accuracy_epoch+1}")
        print(f"MP train: Lowest Train Loss {lowest_train_loss:.6f} at Epoch {lowest_train_loss_epoch+1}")

    def train_plot(self, plot_path: str = utils.DEFAULT_PLOT_PATH):
        # Plot metrics
        plt.figure(figsize=(12, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.epochs + 1), self.train_loss_history, 'b', label='Train Loss')
        plt.plot(range(1, self.epochs + 1), self.val_loss_history, 'r', label='Validation Loss')
        plt.title('Train vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.epochs + 1), self.train_accuracy_history, 'b', label='Train Accuracy')
        plt.plot(range(1, self.epochs + 1), self.val_accuracy_history, 'r', label='Validation Accuracy')
        plt.title('Train vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        plt.tight_layout()

        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
        plt.savefig(plot_path)
        print(f"MP train: Plot saved to {plot_path}")

    def train_metrics(self, epoch: int, epoch_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float):
        print(f"MP train: Epoch {epoch+1}/{self.epochs}, "
                  f"Train Loss: {epoch_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}")

    def save_model(self, model_path: str = utils.DEFAULT_MODEL_PATH):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        np.savez(model_path, w1=self.weightH1, b1=self.bias1, w2=self.weightH2, b2=self.bias2, wOut=self.weightOut, bOut=self.biasOut)
        print(f"MP save: Model saved to {model_path}")
    
    def load_model(self, model_path: str = utils.DEFAULT_MODEL_PATH):
        model_data = np.load(model_path)

        self.weightH1 = model_data['w1']
        self.bias1 = model_data['b1']
        self.weightH2 = model_data['w2']
        self.bias2 = model_data['b2']
        self.weightOut = model_data['wOut']
        self.biasOut = model_data['bOut']
        print(f"MP load: Model loaded from {model_path}")

    def predict(self, X_test):
        predictions = []

        # Loop in all the test samples
        for I in range(0, X_test.shape[0]):  
            
            # Input the data 
            x = X_test[I]
            
            # Feed forward
            z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)  # first hidden layer
            z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)  # second hidden layer
            y = utils.sigmoid_data(np.dot(self.weightOut, z2) + self.biasOut)
            
            # Append the prediction; applying threshold at 0.5 for binary classification
            predictions.append(np.heaviside(y - 0.5, 1)[0])
        
        self.predictions = np.array(predictions)
        print("MP predict: Predictions completed")

    def predict_plot(self, Y_test, cm_path: str = utils.DEFAULT_CONFUSION_MATRIX_PATH):
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
        print(f"MP predict: Confusion matrix saved to {cm_path}")

    def predict_metrics(self, Y_test, X_test):
        cm = confusion_matrix(Y_test, self.predictions)

        # Calculate accuracy
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        print(f"MP predict: Accuracy {accuracy:.4f}")

        # Initialize an array to store predicted probabilities
        predicted_probs = np.zeros(Y_test.shape)

        # Calculate predicted probabilities (feed-forward through the model)
        for i in range(X_test.shape[0]):
            x = X_test[i]

            # Perform forward propagation
            z1 = utils.relu_data(np.dot(self.weightH1, x) + self.bias1)
            z2 = utils.relu_data(np.dot(self.weightH2, z1) + self.bias2)
            y_pred = utils.sigmoid_data(np.dot(self.weightOut, z2) + self.biasOut)

            # Store the predicted probabilities
            predicted_probs[i] = y_pred

        # Calculate binary cross-entropy error
        bce_error = utils.bce_data(Y_test, predicted_probs)
        print(f"MP predict: Binary Cross-Entropy Error {bce_error:.6f}")

    def save_predictions(self, path: str = utils.DEFAULT_PREDICTIONS_PATH):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        labelled_predictions = np.where(self.predictions == 1, 'M', 'B')
        np.savetxt(utils.DEFAULT_PREDICTIONS_PATH, labelled_predictions, delimiter=",", comments='', fmt="%s")
        print(f"MP predict: Predictions saved to {utils.DEFAULT_PREDICTIONS_PATH}")
