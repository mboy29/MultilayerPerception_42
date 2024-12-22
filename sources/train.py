import sources.utils as utils
    
# https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron?select=test.csv
# Training function

def train(train_path, 
          layers_size: list = utils.DEFAULT_LAYERS, 
          eta: float = utils.DEFAULT_LEARNING_RATE,
          epochs: int = utils.DEFAULT_EPOCHS,
          batch_size: int = utils.DEFAULT_BATCH_SIZE,
          decay: float = utils.DEFAULT_DECAY):
    """
    Trains the model using the training data.
    """
    try:
        model = utils.MLP(train_path, layers_size, eta, epochs, batch_size, decay)
        model.train()
        model.train_plot()
        model.save_model()
        
    except Exception as e:
        raise Exception(f"train: error: {e}")

