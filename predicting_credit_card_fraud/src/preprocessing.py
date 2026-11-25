from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# Utils
from utils.logger_utils import logger
# Data Functions
from data import read_dataset, filter_by_data_types

_CURRENT_PATH  = Path(__file__).parent.parent
_DATA_PATH     = _CURRENT_PATH / "data"
_RAW_DATA_PATH = _DATA_PATH / "raw"
_TRAIN_DATASET = _RAW_DATA_PATH / "train.csv"
_TEST_DATASET  = _RAW_DATA_PATH / "test.csv"

def create_onehotencoder() -> OneHotEncoder:
    """Creates OneHotEncoder Object
            Args:
                return(OneHotEncoder): OneHotEncoder object
    """
    return OneHotEncoder(
        categories="auto",
        handle_unknown="ignore",
        drop=None,
        sparse_output=True,
        min_frequency=None,
        dtype=float
    )

def preprocess(
    train_file: Path,
    test_file: Path
    ):
    df = read_dataset(train_file)

    # Drop unimportant columns
    df.drop(
        columns=[
            "account_balance_usd",
            "time_of_day_hour",
            "day_of_week"
        ],
        errors="ignore"
    )

    numerical_columns = filter_by_data_types(data=df, data_type="number")
    string_columns    = filter_by_data_types(data=df, data_type="string")

    numerical_columns.drop(
        columns=[
            "prior_disputes",
            "cvv_retry_count"
        ],
        errors="ignore"
    )