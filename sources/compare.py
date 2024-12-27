import sources.utils as utils

def compare_init_config_input(prompt: str, default_value):
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


def compare_init_config(train_path: str, copy_from: utils.MLP = None) -> utils.MLP:
    """
    Configures and returns an MLP model based on user inputs.

    Parameters:
        train_path (str): The path to the training dataset.
    Returns:
        utils.MLP: The configured MLP model.
    """
    layers = list(map(int, compare_init_config_input(
        f"{utils.BOLD_INFO}Enter the number of units in each hidden layer separated by commas (default {','.join(map(str, utils.DEFAULT_LAYERS))}): {utils.NC}",
        ','.join(map(str, utils.DEFAULT_LAYERS))
    ).split(',')))

    eta = float(compare_init_config_input(
        f"{utils.BOLD_INFO}Enter the learning rate (default {utils.DEFAULT_LEARNING_RATE}): {utils.NC}",
        str(utils.DEFAULT_LEARNING_RATE)
    ))

    epochs = int(compare_init_config_input(
        f"{utils.BOLD_INFO}Enter the number of epochs (default {utils.DEFAULT_EPOCHS}): {utils.NC}",
        str(utils.DEFAULT_EPOCHS)
    ))

    batch_size = int(compare_init_config_input(
        f"{utils.BOLD_INFO}Enter the batch size (default {utils.DEFAULT_BATCH_SIZE}): {utils.NC}",
        str(utils.DEFAULT_BATCH_SIZE)
    ))

    regulation = float(compare_init_config_input(
        f"{utils.BOLD_INFO}Enter the L2 regularization parameter (default {utils.DEFAULT_REGULATION}): {utils.NC}",
        str(utils.DEFAULT_REGULATION)
    ))

    early_stopping = compare_init_config_input(
        f"{utils.BOLD_INFO}Enable early stopping? (no/yes): {utils.NC}",
        'no'
    ).strip().lower() == 'yes'

    patience = int(compare_init_config_input(
        f"{utils.BOLD_INFO}Enter patience (default {utils.DEFAULT_PATIENCE}): {utils.NC}",
        str(utils.DEFAULT_PATIENCE)
    )) if early_stopping else 0

    return utils.MLP(data_path=train_path, layers=layers, eta=eta, epochs=epochs, batch_size=batch_size, 
                     regulation=regulation, early_stopping=early_stopping, patience=patience, copy_from=copy_from)

def compare_init(train_path: str):
    """
    Trains multiple models and stores them in the `models` list, then plots training curves.

    Parameters:
        models (list): An empty list to store the trained models.
        train_path (str): The path to the training dataset.
    Returns: None
    """
    models = []

    while True:
        print(f"{utils.BOLD_INFO}\nMP compare: {utils.INFO}Setting up a new model{utils.NC}")
        if len(models) > 0:
            model = compare_init_config(train_path, models[0])
        else:
            model = compare_init_config(train_path)
        print(model)
        models.append(model)
        print(f"{utils.BOLD_INFO}MP compare: {utils.INFO}New models successfully initiated.{utils.NC}")

        if input(f"\n{utils.BOLD_INFO}Do you want to add another model? (no/yes): {utils.NC}").strip().lower() != 'yes':
            break
    return models

def compare(train_path: str = utils.DEFAULT_TRAIN_PATH, test_path: str = utils.DEFAULT_TEST_PATH, train: bool = None, predict: bool = None):
    """
    Compares the training and prediction functions of multiple MLP models.

    Parameters:
        train_path (str): The path to the training dataset, default to `DEFAULT_TRAIN_PATH`.
        test_path (str): The path to the testing dataset, default to `DEFAULT_TEST_PATH`.
    Returns: None
    """
    try:
        models = compare_init(train_path)
        if train is None and predict is None:
            train = predict = True
        if train:
            utils.MLP.train_compare(models)
        if predict:
            if not train or train is None:
                utils.other_suppress_out()
                utils.MLP.train_compare(models, verbose=True)
                utils.other_restore_out()
            utils.MLP.predict_compare(models, test_path)

    except (KeyboardInterrupt, SystemExit, EOFError):
        print(f"\n{utils.BOLD_INFO}MP compare: {utils.INFO}Goodbye!{utils.NC}")
    except Exception as e:
        print(f"\n{utils.ERROR}MP compare: error: {e}{utils.NC}")
