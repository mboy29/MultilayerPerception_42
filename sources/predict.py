import sources.utils as utils


def predict(test_path: str) -> None:
    try:
        # Load the saved model parameters
        model = utils.MLP()
        model.load_model()

        # Load the test dataset
        data_test = utils.Dataset("test", test_path)
        X_test = data_test.features
        Y_test = data_test.labels

        model.predict(X_test)
        model.predict_plot(Y_test)
        model.predict_metrics(Y_test, X_test)
        model.save_predictions()

    except Exception as e:
        print(f"Error during prediction: {e}")


