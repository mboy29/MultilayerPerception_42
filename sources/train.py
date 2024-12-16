import numpy as np
import sources.utils as utils
    
def train(train, val):
    try:
        print(f"MP train: Loading training and validation data")
        data_train = utils.Dataset("train", train)
        val_data = utils.Dataset("validation", val)
        print(data_train)
        print(val_data)

    except Exception as e:
        raise Exception(f"split: error: {e}")
