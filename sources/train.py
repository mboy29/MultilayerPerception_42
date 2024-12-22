import sources.utils as utils
    
# https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron?select=test.csv
# Training function

def train(train_path: str = utils.DEFAULT_TRAIN_PATH,
          layers_size: list = utils.DEFAULT_LAYERS, 
          eta: float = utils.DEFAULT_LEARNING_RATE,
          epochs: int = utils.DEFAULT_EPOCHS,
          batch_size: int = utils.DEFAULT_BATCH_SIZE,
          decay: float = utils.DEFAULT_REGULATION,
          early_stopping: bool = utils.DEFAULT_EARLY_STOPPING,
          patience: int = utils.DEFAULT_PATIENCE) -> None:
    """
    Trains the model using the training data.
    """
    try:
        model = utils.MLP(train_path, layers_size, eta, epochs, batch_size, decay, early_stopping, patience)
        model.train()
        model.train_plot()
        model.save_model()
        
    except Exception as e:
        raise Exception(f"train: error: {e}")

