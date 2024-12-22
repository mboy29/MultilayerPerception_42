import argparse
import sources as utils

def main():
    args = utils.parse()
    if args.command == "split":
        utils.split(args.data_path, args.ratio)
    elif args.command == "train":
        utils.train(args.data_path, args.layers_size, args.learning_rate, args.epochs, args.batch_size, args.decay)
    elif args.command == "predict":
        utils.predict(args.data_path)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    try:
        print(utils.HEADER)
        main()
    except Exception as e:
        print(f"{utils.ERROR}MP error: {e}")
