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
          patience: int = utils.DEFAULT_PATIENCE,
          nestrov: bool = utils.DEFAULT_NESTEROV,
          momentum: float = utils.DEFAULT_MOMENTUM,
          metrics: bool = utils.DEFAULT_METRICS) -> None:
    """
    Trains the model using the training data.

    Parameters:
        train_path (str): The path to the training data CSV file.
        layers_size (list): The size of the 2 hidden layers.
        eta (float): The learning rate of the model.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size to train the model.
        decay (float): The decay of the model.
        early_stopping (bool): Enable early stopping.
        patience (int): The patience for early stopping.
        nestrov (bool): Enable Nesterov momentum.
        momentum (float): The momentum for Nesterov.
        metrics (bool): Display additional metrics during and after training.
    Returns: None
    """
    try:
        model = utils.MLP(train_path, layers_size, eta, epochs, batch_size, decay, 
                          early_stopping, patience, 
                          nestrov, momentum,
                          metrics)
        model.train()
        model.train_plot()
        model.save_model()
        
    except Exception as e:
        raise Exception(f"train: error: {e}")

