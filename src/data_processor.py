from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [str(BASE_DIR), str(SRC_DIR), ".", os.path.abspath(".")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
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
    from config import (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        LEGACY_RAW_DATA_PATH,
        LEGACY_PROCESSED_DATA_PATH,
        SCALER_PATH,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

logger = logging.getLogger("NeoWatch.DataProcessor")


class DataProcessor:
    """
    Modular data cleaning, transformation, scaling, and SMOTE pipeline
    as specified in NASA Proje Plani 2 (Aşama 2).
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.fitted = False
        self.iqr_bounds: Dict[str, Tuple[float, float]] = {}

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw asteroid dataframe and resolve multicollinearity
        by calculating 'estimated_diameter_mean_km' from min and max diameters.
        """
        df_clean = df.copy()

        # Multi-collinearity resolution: average of min and max diameter
        if "estimated_diameter_mean_km" not in df_clean.columns or df_clean["estimated_diameter_mean_km"].isnull().all():
            if "estimated_diameter_min_km" in df_clean.columns and "estimated_diameter_max_km" in df_clean.columns:
                df_clean["estimated_diameter_mean_km"] = (
                    df_clean["estimated_diameter_min_km"] + df_clean["estimated_diameter_max_km"]
                ) / 2.0

        # Drop rows missing essential features or target
        required_cols = [col for col in FEATURE_COLUMNS if col in df_clean.columns] + [TARGET_COLUMN]
        df_clean = df_clean.dropna(subset=required_cols).reset_index(drop=True)

        # Ensure target is binary integer [0, 1]
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(int)

        logger.info("Cleaned dataset: %d valid records, %d features", len(df_clean), len(FEATURE_COLUMNS))
        return df_clean

    def calculate_iqr_bounds(self, df: pd.DataFrame, factor: float = 3.0) -> Dict[str, Tuple[float, float]]:
        """
        Calculate IQR bounds for numerical features.
        Uses factor=3.0 (relaxed upper bound) to accommodate astronomical scale.
        """
        bounds = {}
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                q25 = float(df[col].quantile(0.25))
                q75 = float(df[col].quantile(0.75))
                iqr = q75 - q25
                lower_bound = max(0.0, q25 - (factor * iqr))
                upper_bound = q75 + (factor * iqr)
                bounds[col] = (lower_bound, upper_bound)
        self.iqr_bounds = bounds
        return bounds

    def treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap extreme non-physical outliers to the 99.5th percentile
        to reduce model leverage while retaining physical validity.
        """
        df_treated = df.copy()
        for col in FEATURE_COLUMNS:
            if col in df_treated.columns:
                cap_val = float(df_treated[col].quantile(0.995))
                df_treated[col] = np.clip(df_treated[col], a_min=0.0, a_max=cap_val)
        return df_treated

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training features and transform."""
        X_feats = X[FEATURE_COLUMNS].copy()
        X_scaled = self.scaler.fit_transform(X_feats)
        self.fitted = True
        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler without data leakage."""
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform or load_scaler.")
        X_feats = X[FEATURE_COLUMNS].copy()
        return self.scaler.transform(X_feats)

    def save_scaler(self, filepath: Path = SCALER_PATH) -> None:
        """Save fitted scaler to models/scaler.pkl (Checkpoint 2)."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info("Saved fitted scaler to %s", filepath)

    def load_scaler(self, filepath: Path = SCALER_PATH) -> None:
        """Load fitted scaler from disk."""
        self.scaler = joblib.load(filepath)
        self.fitted = True
        logger.info("Loaded scaler from %s", filepath)

    def prepare_datasets(
        self,
        raw_csv_path: Optional[Path] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        apply_smote: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        End-to-End Aşama 2 Data Processing Workflow:
        1. Clean raw data & resolve multicollinearity
        2. Treat extreme outliers via IQR capping
        3. Stratified Train/Test split (no data leakage)
        4. Fit StandardScaler on train split & serialize to models/scaler.pkl
        5. Apply SMOTE oversampling to balance minority class
        6. Save processed dataset to data/processed_data.csv (and legacy alias)
        """
        # Determine raw data file path
        if raw_csv_path is None:
            if RAW_DATA_PATH.exists():
                target_path = RAW_DATA_PATH
            elif LEGACY_RAW_DATA_PATH.exists():
                target_path = LEGACY_RAW_DATA_PATH
            else:
                raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH} or {LEGACY_RAW_DATA_PATH}")
        else:
            target_path = Path(raw_csv_path)

        df_raw = pd.read_csv(target_path)
        df_clean = self.clean_raw_data(df_raw)
        df_clean = self.treat_outliers(df_clean)

        X = df_clean[FEATURE_COLUMNS]
        y = df_clean[TARGET_COLUMN].values

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info("Train split: %d samples (Hazardous: %d / Safe: %d)", len(X_train), sum(y_train == 1), sum(y_train == 0))
        logger.info("Test split : %d samples (Hazardous: %d / Safe: %d)", len(X_test), sum(y_test == 1), sum(y_test == 0))

        # Scale features strictly fitted on train set
        X_train_scaled = self.fit_transform(X_train)
        X_test_scaled = self.transform(X_test)
        self.save_scaler()

        # Apply SMOTE to balance minority class (up to 60% ratio)
        if apply_smote:
            smote = SMOTE(random_state=random_state, sampling_strategy=0.6)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            logger.info("After SMOTE -> Train: %s (Hazardous: %d / Safe: %d)", X_train_final.shape, sum(y_train_final == 1), sum(y_train_final == 0))
        else:
            X_train_final, y_train_final = X_train_scaled, y_train

        # Save processed dataset to both Plan 2 and legacy locations
        df_processed = df_clean.copy()
        df_processed[FEATURE_COLUMNS] = self.transform(df_clean[FEATURE_COLUMNS])
        
        df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
        df_processed.to_csv(LEGACY_PROCESSED_DATA_PATH, index=False)
        logger.info("Saved processed dataset to %s and %s", PROCESSED_DATA_PATH, LEGACY_PROCESSED_DATA_PATH)

        return X_train_final, y_train_final, X_test_scaled, y_test, df_clean


# Alias for backward compatibility
AsteroidPreprocessor = DataProcessor
