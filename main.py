import argparse
import sources as utils

def main():
    """
    Main function to parse the arguments and call the appropriate function.

    Parameters: None
    Returns: None
    """
    args = utils.parse()
    if args.command == "split":
        utils.split(args.data_path, args.ratio)
    elif args.command == "train":
        utils.train(args.data_path, args.layers_size, args.learning_rate, args.epochs, args.batch_size, args.regulation, 
                    args.early_stopping, args.patience, 
                    args.nesterov, args.momentum,
                    args.metrics)
    elif args.command == "predict":
        utils.predict(args.data_path, args.confusion_matrix)
    elif args.command == "compare":
        utils.compare(args.data_train_path, args.data_predict_path, args.train_only, args.predict_only)
    elif args.command == "clean":
        utils.clean(args.train, args.predict, args.compare, args.datasets)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    try:
        print(utils.HEADER)
        main()
    except Exception as e:
        print(f"{utils.ERROR}MP error: {e}")
