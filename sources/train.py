import sources.utils as utils
    
# https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron?select=test.csv
# Training function

def train(train_path):
    try:
        print(f"MP train: Loading training and validation data")

        print(f"MP train: Training the model")
        # Train the model, plot the results, and save the model
        model = utils.MLP(train_path)
        model.train()
        model.train_plot()
        model.save_model()
        
    except Exception as e:
        raise Exception(f"train: error: {e}")

