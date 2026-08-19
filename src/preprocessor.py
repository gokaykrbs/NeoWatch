"""
NeoWatch - Feature Engineering & Preprocessing Pipeline
Wrapper around src/data_processor.py for full backward and forward compatibility.
"""

from src.data_processor import DataProcessor, AsteroidPreprocessor
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    LEGACY_RAW_DATA_PATH,
    LEGACY_PROCESSED_DATA_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

__all__ = ["DataProcessor", "AsteroidPreprocessor"]
