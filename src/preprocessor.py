from __future__ import annotations

import os
import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [str(BASE_DIR), str(SRC_DIR), ".", os.path.abspath(".")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
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
except (ImportError, ModuleNotFoundError):
    from data_processor import DataProcessor, AsteroidPreprocessor
    from config import (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        LEGACY_RAW_DATA_PATH,
        LEGACY_PROCESSED_DATA_PATH,
        SCALER_PATH,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

__all__ = ["DataProcessor", "AsteroidPreprocessor"]
