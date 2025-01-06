import sources.utils as utils

def predict(test_path: str = utils.DEFAULT_TEST_PATH, cm: bool = False) -> None:
    """
    Predicts the labels of the test data using the saved model.

    Parameters:
        test_path (str): The path to the test data CSV file.
        cm (bool): If True, a confusion matrix plot is generated.

    Returns: None
    """
    try:
        # Load the saved model parameters
        model = utils.MLP.load_model()

        # Load the test data
        data = utils.read_csv(test_path)
        data_test = utils.Dataset("test", data)
        
        # Predict the test data
        model.predict(test_path)
        if cm:
            model.predict_plot()
        model.save_predictions()

    except Exception as e:
        print(f"Error during prediction: {e}")


