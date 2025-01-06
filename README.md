# MULTILAYER PERCEPTRON

A Python implementation of a Multilayer Perceptron (MLP) that supports customizable configurations, including hyperparameters, training, evaluation, and comparison of models.

## SetUp

Before running the program please make sure you activate the virtual environnment install all the necessary dependencies by running the following command :
```
source ./virtualenv.sh [--activate] [--deactivate]
 - Use --activate or no argument to activate virtual env.
 - Use --deactivate to deactivate virtual env.
```

## Usage

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

## Fundamental Concepts 

### Multilayer Perceptron (MLP)

A Multilayer Perceptron is a type of neural network that consists of:
- **Input Layer**: Accepts input features.
- **Hidden Layers**: Processes data with weights, biases, and activations.
- **Output Layer**: Produces predictions.

Each neuron computes:
- z = W ⋅ X + b
- a = Activation(z)

---

### Core Concepts in MLP Training

#### Feedforwarding
- Data flows **forward** through the network: Input → Hidden Layers → Output.
- Produces predictions: y_pred = Softmax(W ⋅ X + b)


#### Backpropagation
- Adjusts weights and biases by propagating **errors backward**.
- Uses the gradient of the loss with respect to weights to optimize parameters.

#### Gradient Descent
- A method to minimize the **loss function** (error between predictions and labels): θ ← θ - η ⋅ ∇J(θ)
    - `θ`: Parameters (weights and biases).
    - `η`: Learning rate.
    - `∇J(θ)`: Gradient of the loss function  

---

### Matrices in MLP

A **matrix** is a mathematical structure that represents data in a grid of rows and columns. In the context of MLP, matrices are used to represent sets of weights and inputs, allowing for efficient computations in the forward and backward passes.

1. **Weights Matrix**: Represents connections between neurons, with dimensions based on the number of inputs and outputs. Each entry in the matrix is a weight that adjusts the strength of the connection between two neurons.
  
2. **Bias Vector**: A bias vector is a set of values added to the weighted sum at each neuron, shifting the output of the neuron and allowing the network to learn more complex patterns.

3. **Forward Propagation**:
    - **z = W ⋅ X + b**: This represents the weighted sum of the inputs (`X`) with the weights (`W`) and the bias (`b`).
    - **a = Activation(z)**: The activation function (like ReLU or Sigmoid) is applied to the weighted sum to introduce non-linearity.

4. **Backpropagation**: Uses gradients (calculated using matrices) to update the weights and biases to minimize the loss and improve the model.

---

### Key Training Concepts

#### A Batch While Training
- A **batch** is a small subset of the training data used in one training step.
- In `_train_batch`:
1. Perform **forward propagation** to compute predictions.
2. Compute **loss and gradients**.
3. Update **weights and biases**.

#### An Epoch While Training
- An **epoch** processes the entire dataset once.
- In `_train_epoch`:
1. The dataset is split into mini-batches.
2. Each batch is trained sequentially.
3. After processing all batches, compute:
  - **Average loss**.
  - **Accuracy** across all batches.

#### L2 Regularization
- Prevents overfitting by penalizing large weights: J_reg = J_loss + λ ⋅ Σ(W^2)
    - `λ`: Regularization strength.

#### Early Stopping / Patience
- Stops training when the **validation loss** stops improving for a set number of epochs (patience).
- Prevents overfitting by halting at the optimal point.

### Evaluation and Loss Functions

#### Binary Cross-Entropy (BCE)
- A loss function for binary classification tasks: BCE = -1/N Σ [y_i ⋅ log(ŷ_i) + (1 - y_i) ⋅ log(1 - ŷ_i)]
- Penalizes incorrect predictions more heavily.

## Softmax vs. Sigmoid
1. **Sigmoid**:
    - Maps values to probabilities in the range `[0, 1]`.
    - Used for **binary classification**.
    ```
    σ(x) = 1 / (1 + e^(-x))
    ```
2. **Softmax**:
    - Converts outputs to probabilities that **sum to 1**.
    - Used for **multi-class classification**.
    ```
    Softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
    ```


---

### Advanced Optimization Techniques

#### Overfitting
- Happens when the model learns the training data too well (including noise) and fails to generalize.
- **Symptoms**:
- High accuracy on training data.
- Poor accuracy on validation/testing data.
- **Mitigation**:
- Use regularization (e.g., L2 regularization).
- Apply early stopping.
- Add more data or use data augmentation.

#### Nesterov Momentum
- An improvement over standard momentum by adding a **lookahead step**:
1. **Look ahead** to estimate the position:
   ```
   θ_lookahead = θ + μ ⋅ v_(t-1)
   ```
2. Compute gradient at this position:
   ```
   v_t = μ ⋅ v_(t-1) - η ⋅ ∇J(θ_lookahead)
   ```
3. Update the weights:
   ```
   θ ← θ + v_t
   ```
- **Benefit**: Helps converge faster and avoids overshooting.