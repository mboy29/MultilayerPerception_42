import numpy as np
import matplotlib.pyplot as plt
import sources.utils as utils
from sklearn.model_selection import train_test_split
    
# https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron?select=test.csv
# Training function

def ANN_train(X_train, Y_train, X_val, Y_val, p=24, q=24, eta=0.0314, epochs=84, batch_size=8):
    """
    Trains an artificial neural network using backpropagation and computes loss and accuracy for training and validation data.

    Parameters:
        X_train (np.ndarray): Training features.
        Y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        Y_val (np.ndarray): Validation labels.
        p (int): Number of neurons in the first hidden layer.
        q (int): Number of neurons in the second hidden layer.
        eta (float): Learning rate.
        epochs (int): Number of training epochs.
        batch_size (int): Size of mini-batches.

    Returns:
        tuple: Trained weights, biases, loss and accuracy history for training and validation.
    """

    # Initialize weights and biases
    w1 = 2 * np.random.rand(p, X_train.shape[1]) - 0.5  # Layer 1 weights
    b1 = np.random.rand(p)

    w2 = 2 * np.random.rand(q, p) - 0.5                # Layer 2 weights
    b2 = np.random.rand(q)

    wOut = 2 * np.random.rand(q) - 0.5                 # Output layer weights
    bOut = np.random.rand(1)

    # Histories for loss and accuracy
    train_loss_history = []  
    train_accuracy_history = []  
    val_loss_history = []    
    val_accuracy_history = []  

    # Training over epochs
    for epoch in range(epochs):
        # Initialize metrics for this epoch
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Training Loop (Mini-batch Gradient Descent)
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            batch_loss = 0
            for j in range(len(X_batch)):
                # Forward propagation
                x = X_batch[j]
                z1 = utils.relu_data(np.dot(w1, x) + b1)       # Layer 1
                z2 = utils.relu_data(np.dot(w2, z1) + b2)      # Layer 2
                y_pred = utils.sigmoid_data(np.dot(wOut, z2) + bOut)  # Output layer

                # Compute predictions and accuracy
                y_pred_label = 1 if y_pred >= 0.5 else 0
                if y_pred_label == Y_batch[j]:
                    correct_predictions += 1
                total_predictions += 1

                # Compute error
                delta_Out = 2 * (y_pred - Y_batch[j]) * utils.sigmoid_data(y_pred, der=True)

                # Backpropagation
                delta_2 = delta_Out * wOut * utils.relu_data(z2, der=True)
                delta_1 = np.dot(delta_2, w2) * utils.relu_data(z1, der=True)

                # Gradient updates
                wOut -= eta * delta_Out * z2
                bOut -= eta * delta_Out

                w2 -= eta * np.outer(delta_2, z1)
                b2 -= eta * delta_2

                w1 -= eta * np.outer(delta_1, x)
                b1 -= eta * delta_1

                # Accumulate batch loss
                batch_loss += ((y_pred - Y_batch[j]) ** 2).item()

            # Average loss for the batch
            epoch_loss += batch_loss / len(X_batch)

        # Training loss and accuracy for the epoch
        epoch_loss /= (X_train.shape[0] / batch_size)
        train_accuracy = correct_predictions / total_predictions
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(train_accuracy)

        # Validation Loss and Accuracy
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0

        for i in range(X_val.shape[0]):
            x = X_val[i]
            y_true = Y_val[i]

            # Forward propagation on validation set
            z1 = utils.relu_data(np.dot(w1, x) + b1)
            z2 = utils.relu_data(np.dot(w2, z1) + b2)
            y_pred = utils.sigmoid_data(np.dot(wOut, z2) + bOut)

            # Compute loss
            val_loss += ((y_pred - y_true) ** 2).item()

            # Compute predictions and accuracy
            y_pred_label = 1 if y_pred >= 0.5 else 0
            if y_pred_label == y_true:
                val_correct_predictions += 1
            val_total_predictions += 1

        val_loss /= X_val.shape[0]
        val_accuracy = val_correct_predictions / val_total_predictions

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {epoch_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}")

    # Plot metrics
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss_history, 'b', label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss_history, 'r', label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracy_history, 'b', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracy_history, 'r', label='Validation Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return w1, b1, w2, b2, wOut, bOut, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history


def train(train_path, val_path):
    try:
        print(f"MP train: Loading training and validation data")
        
        # Load datasets
        data_train = utils.Dataset("train", train_path)
        data_val = utils.Dataset("validation", val_path)
        
        # Extract features and labels directly
        X_train = data_train.features
        Y_train = data_train.labels
        X_val = data_val.features
        Y_val = data_val.labels

        print("Training Class Distribution:", np.bincount(Y_train))
        print("Validation Class Distribution:", np.bincount(Y_val))

        # Train the model
        w1, b1, w2, b2, wOut, bOut, train_loss, train_accuracy, val_loss, val_accuracy = ANN_train(X_train, Y_train, X_val, Y_val)

        # Save the model
        np.savez(utils.DEFAUlT_MODEL_PATH, w1=w1, b1=b1, w2=w2, b2=b2, wOut=wOut, bOut=bOut)
    except Exception as e:
        raise Exception(f"train: error: {e}")

