from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# Utils
from utils.logger_utils import logger
# Preprocess functions
from preprocessing import preprocess

_CURRENT_PATH  = Path(__file__).parent.parent
_DATA_PATH     = _CURRENT_PATH / "data"
_RAW_DATA_PATH = _DATA_PATH / "raw"
_TRAIN_DATASET = _RAW_DATA_PATH / "train.csv"
_TEST_DATASET  = _RAW_DATA_PATH / "test.csv"

def train(
    train_file: Path,
    test_file: Path
    ):
    """
    """



if __name__ == "__main__":
    train(_TRAIN_DATASET, _TEST_DATASET)