import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Optional for heatmap visualization
import sources.utils as utils

def ANN_pred(X_test, w1, b1, w2, b2, wOut, bOut, mu):
    pred = []
    for I in range(0, X_test.shape[0]):  # loop in all the test samples
        # 1: input the data 
        x = X_test[I]
        
        # 2.1: Feed forward
        z1 = utils.relu_data(np.dot(w1, x) + b1)  # first hidden layer
        z2 = utils.relu_data(np.dot(w2, z1) + b2)  # second hidden layer
        y = utils.sigmoid_data(np.dot(wOut, z2) + bOut)  # output layer
        
        # Append the prediction; applying threshold at 0.5 for binary classification
        pred.append(np.heaviside(y - 0.5, 1)[0])
    
    return np.array(pred)

def predict(test_path: str) -> None:
    try:
        # Load the saved model parameters
        model_data = np.load(utils.DEFAUlT_MODEL_PATH)
        w1 = model_data['w1']
        b1 = model_data['b1']
        w2 = model_data['w2']
        b2 = model_data['b2']
        wOut = model_data['wOut']
        bOut = model_data['bOut']
        
        # Load the test dataset
        data_test = utils.Dataset("test", test_path)
        X_test = data_test.features
        Y_test = data_test.labels

        # Perform predictions using the ANN_pred function
        predictions = ANN_pred(X_test, w1, b1, w2, b2, wOut, bOut, mu=None)  # mu can be passed if needed

        # Print predictions
        print("Predictions:")
        print(predictions)

        # Save the predictions to a file if required
        np.savetxt("predictions.csv", predictions, delimiter=",", header="Predictions", comments='')

        # Calculate confusion matrix
        cm = confusion_matrix(Y_test, predictions)
        
        # Print the confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # Calculate accuracy
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        print(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Error during prediction: {e}")
