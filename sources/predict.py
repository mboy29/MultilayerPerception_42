import sources.utils as utils

def predict(test_path: str, cm: bool = False) -> None:
    try:
        # Load the saved model parameters
        model = utils.MLP()
        model.load_model()

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


