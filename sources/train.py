import numpy as np
import matplotlib.pyplot as plt
import sources.utils as utils
from sklearn.model_selection import train_test_split
    
# https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron?select=test.csv
# Training function

def train(train_path, val_path):
    try:
        print(f"MP train: Loading training and validation data")
        
        # Load datasets
        train_data = utils.Dataset("train", train_path)
        val_data = utils.Dataset("validation", val_path)

        # Train the model, plot the results, and save the model
        model = utils.MLP(train_data, val_data)
        model.train()
        model.train_plot()
        model.save_model()
        
    except Exception as e:
        raise Exception(f"train: error: {e}")

