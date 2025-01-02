import sources.utils as utils

def clean(train: bool, predict: bool, compare: bool, datasets: bool) -> None:
    """
    Clean the training, prediction, and comparison files and directories as well as the training and testing datasets.

    Parameters:
        train (bool): Whether to clean the training files.
        predict (bool): Whether to clean the prediction files.
        compare (bool): Whether to clean the comparison files.
        datasets (bool): Whether to clean the training and testing datasets.
    Returns: None
    """
    # If no arguments are provided, clean everything
    if not (train or predict or compare or datasets):
        train = predict = compare = datasets = True

    # Only clean the specified directories
    if train:
        utils.other_delete_directory(utils.DEFAULT_TRAIN_DIR_PATH)
        print(f"{utils.BOLD_INFO}MP clean: {utils.INFO}Removed '{utils.DEFAULT_TRAIN_DIR_PATH}' directory{utils.NC}")
    if predict:
        utils.other_delete_directory(utils.DEFAULT_PREDICT_DIR_PATH)
        print(f"{utils.BOLD_INFO}MP clean: {utils.INFO}Removed '{utils.DEFAULT_PREDICT_DIR_PATH}' directory{utils.NC}")
    if compare:
        utils.other_delete_directory(utils.DEFAULT_COMPARE_DIR_PATH)
        print(f"{utils.BOLD_INFO}MP clean: {utils.INFO}Removed '{utils.DEFAULT_COMPARE_DIR_PATH}' directory{utils.NC}")
    if datasets:
        utils.other_delete_file(utils.DEFAULT_TRAIN_PATH)
        utils.other_delete_file(utils.DEFAULT_TEST_PATH)
        print(f"{utils.BOLD_INFO}MP clean: {utils.INFO}Removed training and testing datasets from '{utils.DEFAULT_DATASETS_DIR_PATH}'{utils.NC}")
