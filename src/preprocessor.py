"""
NeoWatch - Feature Engineering & Preprocessing Pipeline
Wrapper around src/data_processor.py for full backward and forward compatibility.
"""

import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if "." not in sys.path:
    sys.path.insert(0, ".")

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
