import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sources.utils as utils

def configure_input(prompt: str, default_value):
    """
    Get user input and return the default value if the input is empty.

    Parameters:
        prompt (str): The prompt to display to the user.
        default_value: The default value to return if the input is empty.
    Returns:
        str: The user input or the default value.
    """
    user_input = input(prompt).strip()
    return user_input if user_input else default_value


def compare_configure(train_path: str) -> utils.MLP:
    """
    Configures and returns an MLP model based on user inputs.

    Parameters:
        train_path (str): The path to the training dataset.
    Returns:
        utils.MLP: The configured MLP model.
    """
    layers = list(map(int, configure_input(
        f"{utils.BOLD_INFO}Enter the number of units in each hidden layer separated by commas (default {','.join(map(str, utils.DEFAULT_LAYERS))}): {utils.NC}",
        ','.join(map(str, utils.DEFAULT_LAYERS))
    ).split(',')))

    eta = float(configure_input(
        f"{utils.BOLD_INFO}Enter the learning rate (default {utils.DEFAULT_LEARNING_RATE}): {utils.NC}",
        str(utils.DEFAULT_LEARNING_RATE)
    ))

    epochs = int(configure_input(
        f"{utils.BOLD_INFO}Enter the number of epochs (default {utils.DEFAULT_EPOCHS}): {utils.NC}",
        str(utils.DEFAULT_EPOCHS)
    ))

    batch_size = int(configure_input(
        f"{utils.BOLD_INFO}Enter the batch size (default {utils.DEFAULT_BATCH_SIZE}): {utils.NC}",
        str(utils.DEFAULT_BATCH_SIZE)
    ))

    regulation = float(configure_input(
        f"{utils.BOLD_INFO}Enter the L2 regularization parameter (default {utils.DEFAULT_REGULATION}): {utils.NC}",
        str(utils.DEFAULT_REGULATION)
    ))

    early_stopping = configure_input(
        f"{utils.BOLD_INFO}Enable early stopping? (no/yes): {utils.NC}",
        'no'
    ).strip().lower() == 'yes'

    patience = int(configure_input(
        f"{utils.BOLD_INFO}Enter patience (default {utils.DEFAULT_PATIENCE}): {utils.NC}",
        str(utils.DEFAULT_PATIENCE)
    )) if early_stopping else 0

    return utils.MLP(data_path=train_path, layers=layers, eta=eta, epochs=epochs, batch_size=batch_size, 
                     regulation=regulation, early_stopping=early_stopping, patience=patience)

def compare_train_plot(models: list, plot_path: str = utils.DEFAULT_COMPARE_PLOT_PATH):
    """
    Plots training and validation curves for all models.

    Parameters:
        models (list): A list of trained models.
        plot_path (str): The path to save the plot, default to `DEFAULT_COMPARE_PLOT_PATH`.
    Returns: None
    """
    plt.figure(figsize=(14, 7))

    # Plot Loss and Accuracy curves
    for i, model in enumerate(models):
        epochs_range = range(len(model.train_loss_history))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, model.train_loss_history, label=f'Model {i + 1} Train Loss')
        plt.plot(epochs_range, model.val_loss_history, label=f'Model {i + 1} Validation Loss', linestyle='--')
        plt.title("Train vs Validation Loss")
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, model.train_accuracy_history, label=f'Model {i + 1} Train Accuracy')
        plt.plot(epochs_range, model.val_accuracy_history, label=f'Model {i + 1} Validation Accuracy', linestyle='--')
        plt.title("Train vs Validation Accuracy")

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.savefig(plot_path)
    print(f"{utils.BOLD_INFO}MP compare: {utils.INFO}Plot saved at: {plot_path}{utils.NC}")


def compare_train(models: list, train_path: str):
    """
    Trains multiple models and stores them in the `models` list, then plots training curves.

    Parameters:
        models (list): An empty list to store the trained models.
        train_path (str): The path to the training dataset.
    Returns: None
    """
    while True:
        print(f"{utils.BOLD_INFO}MP compare: {utils.INFO}Setting up a new model{utils.NC}")
        model = compare_configure(train_path)

        # Train the model
        utils.other_suppress_out()
        model.train()
        utils.other_restore_out()

        if models:
            model.train_data = models[0].train_data
            model.val_data = models[0].val_data

        models.append(model)
        print(f"{utils.BOLD_INFO}MP compare: {utils.INFO}Model training complete!{utils.NC}")

        if input(f"\n{utils.BOLD_INFO}Do you want to add another model? (no/yes): {utils.NC}").strip().lower() != 'yes':
            break

    compare_train_plot(models)


def compare_predict_plot(models: list, X_predict: np.ndarray, Y_predict: np.ndarray, plot_path: str = utils.DEFAULT_COMPARE_CONFUSION_MATRIX_PATH):
    """
    Plots confusion matrices for all models.

    Parameters:
        models (list): A list of trained models.
        X_predict (np.ndarray): The features of the testing dataset.
        Y_predict (np.ndarray): The labels of the testing dataset.
        plot_path (str): The path to save the plot.
    Returns: None
    """
    plt.figure(figsize=(10 * len(models), 8))
    
    for i, model in enumerate(models):
        utils.other_suppress_out()
        predictions, _, _ = model.predict(X_predict, Y_predict)
        utils.other_restore_out()

        cm = confusion_matrix(Y_predict, predictions)

        plt.subplot(1, len(models), i + 1)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(Y_predict), yticklabels=np.unique(Y_predict))
        plt.title(f"Model {i + 1}")
        plt.xlabel('Predicted')
        plt.ylabel('True')

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.savefig(plot_path)
    plt.close()
    print(f"{utils.BOLD_INFO}MP compare: {utils.INFO}Confusion matrix plot saved at: {plot_path}{utils.NC}")

def compare_predict(models: list, test_path: str) -> dict:
    """
    Makes predictions using each trained model and tracks their metrics.

    Parameters:
        models (list): A list of trained models.
        test_path (str): The path to the testing dataset.
    Returns: None
    """
    metrics = {}
    data = utils.read_csv(test_path)
    dataset = utils.Dataset("predict", data)
    X_predict, Y_predict = dataset.features, dataset.labels

    print(f"\n{utils.BOLD_INFO}MP compare: {utils.INFO}Making predictions using the trained models{utils.NC}")
    
    for i, model in enumerate(models):
        utils.other_suppress_out()
        predictions, accuracy, loss = model.predict(X_predict, Y_predict)
        utils.other_restore_out()

        metrics[f"Model {i + 1}"] = {"predictions": predictions, "accuracy": accuracy, "loss": loss}

        print(f"{utils.BOLD_INFO}MP compare: {utils.INFO}{i + 1} {model}{utils.NC}")
        print(f"{utils.INFO}  - Accuracy: {accuracy:.2f}{utils.NC}")
        print(f"{utils.INFO}  - Loss: {loss:.4f}{utils.NC}")

    compare_predict_plot(models, X_predict, Y_predict)
    best_model_index = min(metrics, key=lambda x: metrics[x]['loss'])
    print(f"\n{utils.BOLD_INFO}MP compare: {utils.INFO}Model {best_model_index} has the lowest loss: {metrics[best_model_index]['loss']:.4f}{utils.NC}")


def compare(train_path: str = utils.DEFAULT_TRAIN_PATH, test_path: str = utils.DEFAULT_TEST_PATH):
    """
    Compares the training and prediction functions of multiple MLP models.

    Parameters:
        train_path (str): The path to the training dataset, default to `DEFAULT_TRAIN_PATH`.
        test_path (str): The path to the testing dataset, default to `DEFAULT_TEST_PATH`.
    Returns: None
    """
    try:
        models = []

        compare_train(models, train_path)
        compare_predict(models, test_path)

    except (KeyboardInterrupt, SystemExit, EOFError):
        print(f"\n{utils.BOLD_INFO}MP compare: {utils.INFO}Goodbye!{utils.NC}")
    except Exception as e:
        print(f"\n{utils.ERROR}MP compare: error: {e}{utils.NC}")
