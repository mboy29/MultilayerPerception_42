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

### Fundamentals

1. **Multilayer Perceptron (MLP)**:

    A Multilayer Perceptron is a type of neural network that consists of:

    - **Input Layer**: Accepts input features.
    - **Hidden Layers**: Processes data with weights, biases, and activations.
    - **Output Layer**: Produces predictions.

    Each neuron computes:

    - `z = W ⋅ X + b`
    - `a = Activation(z)`

2. **Core Concepts in MLP Training**:

    a. **Feedforwarding**:

    - Data flows **forward** through the network:  
    Input → Hidden Layers → Output.

    - Produces predictions:  
        `y_pred = Softmax(W ⋅ X + b)`

    b. **Backpropagation**:

    - Adjusts weights and biases by propagating **errors backward**.
    
    - Uses the gradient of the loss with respect to weights to optimize parameters.

    c. **Gradient Descent**:
    
    A method to minimize the **loss function** (error between predictions and labels): `θ ← θ - η ⋅ ∇J(θ)`

    - `θ`: Parameters (weights and biases).
    - `η`: Learning rate.
    - `∇J(θ)`: Gradient of the loss function.

3. **Matrices in MLP**:

    a. **Weights Matrix**: 

    - Represents connections between neurons, with dimensions based on the number of inputs and outputs.
   
    b. **Bias Vector**: 
    
    - Shifts the weighted sums for each neuron.
    
    c. **Forward Propagation**:

    - `z = W ⋅ X + b`
    - `a = Activation(z)`

    d. **Backpropagation**: 
    
    - Uses gradients (matrices) to update weights and biases.

4. **Key Training Concepts**:

    a. **Batch**:
    - Small subset of the training data used in one training step.

    b. **Epoch**:
    - Processes the entire dataset once.

    c. **L2 Regularization**:
    
    - Prevents overfitting by penalizing large weights: 
        
        `J_reg = J_loss + λ ⋅ Σ(W^2)`, where `λ` represents the regularization strength.

5. **Evaluation and Loss Functions**

    a. **Binary Cross-Entropy (BCE)**: 
    
    - A loss function for binary classification tasks: 
        
        `BCE = -1/N Σ [y_i ⋅ log(ŷ_i) + (1 - y_i) ⋅ log(1 - ŷ_i)]`

    - Penalizes incorrect predictions more heavily.

    b. **Softmax vs. Sigmoid**:
    
    - **Sigmoid**:

        - Maps values to probabilities in the range `[0, 1]`. Commonly used for **binary classification**: 
            
            `σ(x) = 1 / (1 + e^(-x))`
    
    - **Softmax**: 

        - Converts outputs to probabilities that **sum to 1**. Commonly used for **multi-class classification**: 
        
            `Softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j`

6. **Advanced Optimization Techniques**:

    a. **Overfitting**:

    - Happens when the model learns the training data too well (including noise) and fails to generalize.

    - **Symptoms**:
        - High accuracy on training data.
        - Poor accuracy on validation/testing data.
        
    - **Mitigation**:
        - Use regularization (e.g., L2 regularization).
        - Apply early stopping.
        - Add more data or use data augmentation.

    b. **Nesterov Momentum**:

    - An improvement over standard momentum by adding a **lookahead step**:
        1. **Look ahead** to estimate the position: 
            
            `θ_lookahead = θ + μ ⋅ v_(t-1)`

        2. Compute gradient at this position: 
        
            `v_t = μ ⋅ v_(t-1) - η ⋅ ∇J(θ_lookahead)`

        3. Update the weights: 
            
            `θ ← θ + v_t`

    - **Benefit**: Helps converge faster and avoids overshooting.
