# MULTILAYER PERCEPTRON

A Python implementation of a Multilayer Perceptron (MLP) that supports customizable configurations, including hyperparameters, training, evaluation, and comparison of models.

### SetUp

Before running the program please make sure you activate the virtual environnment install all the necessary dependencies by running the following command :
```
source ./virtualenv.sh [--activate] [--deactivate]
 - Use --activate or no argument to activate virtual env.
 - Use --deactivate to deactivate virtual env.
```

### Usage

The main entry point of this project is main.py. 
```
python3 -B main.py
```

The script supports multiple commands, each with its own set of parameters. Below is a detailed guide to the available commands.

- Split: Split the dataset into training and testing sets

    - `-h`, `--help`: Show a help message
    - `-r`, `--ratio RATIO`: The ratio of data to be included in the training set
    - `-d`, `--data_path DATA_PATH`: Path to the CSV file containing the dataset (default: 'datasets/data.csv')

    ```
    python3 -B main.py split 
        [-h, --help] 
        [-r, --ratio <unfloat>] 
        [-d, --data_path <string>]
    ```

- Train: Train the multilayer perceptron model

    - `-h`, `--help`: Show a help message
    - `-d`, `--data_path DATA_PATH`: Path to the CSV file containing the training set (default: 'datasets/train.csv')
    - `-l`, `--layers_size LAYERS_SIZE`: The size of the 2 hidden layers (e.g.: 4, 4)
    - `-eta`, `--learning_rate LEARNING_RATE`: The learning rate of the model
    - `-e`, `--epochs EPOCHS`: The number of epochs to train the model
    - `-bs`, `--batch_size BATCH_SIZE`: The batch size to train the model
    - `-r`, `--regulation REGULATION`: The decay of the model
    - `-es`, `--early_stopping`: Enable early stopping
    - `-p`, `--patience PATIENCE`: The patience for early stopping
    - `-n`, `--nesterov`: Enable Nesterov momentum
    - `-mo`, `--momentum MOMENTUM`: The momentum for Nesterov
    - `-m`, `--metrics`: Display additional metrics during and after training

    ```
    python3 -B main.py train 
        [-h, --help] 
        [-d, --data_path <string>]
        [-l, --layers_size <list>]
        [-eta, --learning_rate <unfloat>]
        [-bs, --batch_size <unint>]
        [-r, --regulation <unfloat>]
        [-es, --early_stopping]
        [-n, --nesterov]
        [-mo, --momentum <unfloat>]
        [-m, --metrics]
    ```

- Predict: Predict the labels of the testing set using the trained model

    - `-h`, `--help`: Show a help message
    - `-d`, `--data_path DATA_PATH`: Path to the CSV file containing the testing set (default: 'datasets/test.csv')
    - `cm`, `--confusion_matrix`: Create the confusion matrix

    ```
    python3 -B main.py predict 
        [-h, --help] 
        [-d, --data_path <string>]
        [-cm, --confusion_matrix]
    ```

- Compare: Compares multiple models on their training and predictions

    - `-h`, `--help`: Show a help message
    - `-dt`, `--data_train_path DATA_TRAIN_PATH`: Path to the CSV file containing the testing set (default: 'datasets/test.csv')
    - `-dp`, `--data_predict_path DATA_PREDICT_PATH`: Path to the CSV file containing the predictions (default: 'datasets/test.csv')
    - `-t`, `--train_only`: Only compare the models on their training
    - `-p`, `--predict_only`: Only compare the models on their predictions

    ```
    python3 -B main.py clean 
        [-h, --help] 
        [-dt, --data_train_path <string>]
        [-dp, --data_predict_path <string>]
        [-t, --train_only]
        [-p, --predict_only]
    ```

- Clean: Clean the project directory

    - `-h`, `--help`: Show a help message
    - `-t`, `--train`: Remove the training directory
    - `-p`, `--predict`: Remove the predict directory
    - `-c`, `--compare`: Remove the compare directory
    - `-d`, `--datasets`: Remove the train and test datasets

    ```
    python3 -B main.py clean 
        [-h, --help] 
        [-t, --train]
        [-p, --predict]
        [-c, --compare]
        [-d, --datasets]
    ```
