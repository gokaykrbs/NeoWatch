"""
NeoWatch - Configuration & Global Constants
Manages environment variables, API endpoints, directory paths, and project settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"

# Plan 2 Primary Paths (NASA Proje Plani 2.pdf)
RAW_DATA_PATH = DATA_DIR / "raw_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "asteroid_xgb_model.pkl"

# Plan 1 Legacy Aliases for Full Backward Compatibility
LEGACY_RAW_DATA_PATH = DATA_DIR / "raw_asteroid_data.csv"
LEGACY_PROCESSED_DATA_PATH = DATA_DIR / "processed_asteroid_data.csv"
LEGACY_MODEL_PATH = MODELS_DIR / "asteroid_model.pkl"

# Load .env file
load_dotenv(BASE_DIR / ".env")

# NASA NeoWs API Configuration
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
NASA_FEED_BASE_URL = "https://api.nasa.gov/neo/rest/v1/feed"
NASA_NEO_BROWSE_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"

# Rate Limiting & Network Settings
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5

# Feature Definitions
FEATURE_COLUMNS = [
    "absolute_magnitude_h",
    "estimated_diameter_min_km",
    "estimated_diameter_max_km",
    "estimated_diameter_mean_km",
    "relative_velocity_km_s",
    "miss_distance_km",
]

TARGET_COLUMN = "is_potentially_hazardous_asteroid"
