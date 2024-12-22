import os

ERROR='\033[38;2;255;173;194m'
INFO='\033[38;2;188;212;230m'
BOLD_INFO='\033[38;2;153;193;222m'
SUCCESS='\033[38;2;207;225;185m'
NC='\033[0m'


DEFAULT_DATASETS_DIR_PATH = "datasets"
DEFAULT_TRAIN_PATH = os.path.join(DEFAULT_DATASETS_DIR_PATH, "train.csv")
DEFAULT_VALIDATION_PATH = os.path.join(DEFAULT_DATASETS_DIR_PATH, "validation.csv")
DEFAULT_TEST_PATH = os.path.join(DEFAULT_DATASETS_DIR_PATH, "test.csv")
DEFAULT_DATASET_PATH = "datasets/data.csv"

DEFAULT_TRAIN_DIR_PATH = "train"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_TRAIN_DIR_PATH, "model.npz")
DEFAULT_PLOT_PATH = os.path.join(DEFAULT_TRAIN_DIR_PATH, "plot.png")
DEFAULT_SCALER_PATH = os.path.join(DEFAULT_TRAIN_DIR_PATH, "scaler.pkl")

DEFAULT_PREDICT_DIR_PATH = "predict"
DEFAULT_PREDICTIONS_PATH = os.path.join(DEFAULT_PREDICT_DIR_PATH, "predictions.csv")
DEFAULT_CONFUSION_MATRIX_PATH = os.path.join(DEFAULT_PREDICT_DIR_PATH, "confusion_matrix.png")

DEFAULT_RATIO = 0.7
DEFAULT_LAYERS = [8, 8]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 84
DEFAULT_BATCH_SIZE = 8
DEFAULT_REGULATION = 0.0025
DEFAULT_EARLY_STOPPING = False
DEFAULT_PATIENCE = 8

HEADER = f"{BOLD_INFO}\n----------------------------------------------------------------\n" \
        f"{INFO} __  __  __  __  __   ____  ____  __      __   _  _  ____  ____ \n" \
        "(  \\/  )(  )(  )(  ) (_  _)(_  _)(  )    /__\\ ( \\/ )( ___)(  _ \\\n" \
        " )    (  )(__)(  )(__  )(   _)(_  )(__  /(__)\\ \\  /  )__)  )   /\n" \
        "(_/\\/\\_)(______)(____)(__) (____)(____)(__)(__)(__) (____)(_)\\_)\n" \
        "   ____  ____  ____   ___  ____  ____  ____  ____  _____  _  _    \n" \
        "  (  _ \\( ___)(  _ \\ / __)( ___)(  _ \\(_  _)(  _ \\(  _  )( \\( )   \n" \
        "   )___/ )__)  )   /( (__  )__)  )___/  )(   )   / )(_)(  )  (    \n" \
        "  (__)  (____)(_)\\_) \\___)(____)(__)   (__) (_)\\_)(_____)(_)\\_)   \n" \
        f"{BOLD_INFO}\n----------------------------------------------------------- MBOY\n{NC}\n"