import pandas as pd
from pathlib import Path
from typing import Optional, Literal

# Utils
from utils.logger_utils import logger

def read_dataset(file_path: Path) -> Optional[pd.DataFrame | bool]:
    """
    """
    logger.info(f"File '{file_path}' is reading now.\n")
    try:
        df = pd.read_csv(file_path)
        return df.copy()
    except (FileExistsError, FileNotFoundError) as e:
        logger.error(f"File is not found or not exist: {str(e).strip()}")
        return False

def filter_by_data_types(data: pd.DataFrame, data_type: Literal["number", "string"]) -> Optional[pd.DataFrame | bool]:
    """
    """
    if not data_type == "number" or not data == "string":
        logger.error(f"Data type input is wrong. It should be 'number' or 'string'\n")
        raise ValueError("Data type input is wrong. It should be 'number' or 'string'")
    try:
        return data.select_dtypes(include=data_type.strip())
    except Exception as e:
        logger.error(f"Error from data filtering: {str(e).strip()}\n")
        return False