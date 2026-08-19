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

# Default Registered NASA API Key (ensures uninterrupted service in cloud/clean environments)
DEFAULT_REGISTERED_API_KEY = "LudI23zGGE14cnCd8a4D0DhiNuIoYaDBbZOTN8ly"

def get_nasa_api_key(custom_key: str | None = None) -> str:
    """
    Resolve the best available NASA NeoWs API key with multi-source hierarchy:
    1. Direct runtime custom_key parameter (e.g. from UI input)
    2. Streamlit secrets (st.secrets["NASA_API_KEY"])
    3. Environment variable (os.getenv("NASA_API_KEY"))
    4. Verified registered fallback key (DEFAULT_REGISTERED_API_KEY)
    5. DEMO_KEY
    """
    if custom_key and str(custom_key).strip():
        return str(custom_key).strip()

    # Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "NASA_API_KEY" in st.secrets:
            sec_key = str(st.secrets["NASA_API_KEY"]).strip()
            if sec_key and sec_key != "DEMO_KEY":
                return sec_key
    except Exception:
        pass

    # Check environment variable / .env
    env_key = os.getenv("NASA_API_KEY", "").strip()
    if env_key and env_key != "DEMO_KEY":
        return env_key

    # Return registered production key to prevent rate limit blocks on Streamlit Cloud / clean clones
    return DEFAULT_REGISTERED_API_KEY


# NASA NeoWs API Configuration
NASA_API_KEY = get_nasa_api_key()
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
