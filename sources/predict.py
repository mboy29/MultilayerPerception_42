import sources.utils as utils

def predict(test_path: str = utils.DEFAULT_TEST_PATH, cm: bool = False) -> None:
    """
    Predict the labels of the test dataset using the saved model.

    Parameters:
        test_path (str): The path to the test dataset.
        cm (bool): Whether to display the confusion matrix or not.
    Returns: None
    """
    try:
        # Load the saved model parameters
        model = utils.MLP.load_model()

        # Load the test dataset

        data = utils.read_csv(test_path)
        data_test = utils.Dataset("test", data)
        X_test = data_test.features
        Y_test = data_test.labels

        model.predict(X_test, Y_test)
        if cm:
            model.predict_plot(Y_test)
        model.save_predictions()

    except Exception as e:
        print(f"Error during prediction: {e}")


