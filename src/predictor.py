"""
NeoWatch - Real-Time Inference & Risk Scoring Engine
Loads serialized model and scaler artifacts to generate real-time hazard predictions and risk scores.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import joblib

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if "." not in sys.path:
    sys.path.insert(0, ".")

from src.config import (
    SCALER_PATH,
    MODEL_PATH,
    LEGACY_MODEL_PATH,
    FEATURE_COLUMNS,
)

logger = logging.getLogger("NeoWatch.Predictor")


class AsteroidPredictor:
    """Inference engine for predicting asteroid hazard probability."""

    def __init__(self, scaler_path: Optional[Path] = None, model_path: Optional[Path] = None):
        self.scaler_path = scaler_path or SCALER_PATH
        self.model_path = model_path or MODEL_PATH
        self.scaler = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load scaler and model if they exist on disk (with fallback to legacy paths)."""
        try:
            if Path(self.scaler_path).exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded scaler from %s", self.scaler_path)
            else:
                logger.warning("Scaler artifact not found at %s", self.scaler_path)
        except Exception as exc:
            logger.error("Failed to load scaler from %s: %s", self.scaler_path, exc)
            self.scaler = None

        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info("Loaded model from %s", self.model_path)
            elif Path(LEGACY_MODEL_PATH).exists():
                self.model = joblib.load(LEGACY_MODEL_PATH)
                logger.info("Loaded fallback model from %s", LEGACY_MODEL_PATH)
            else:
                logger.warning("Model artifact not found at %s or %s", self.model_path, LEGACY_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", self.model_path, exc)
            self.model = None

    @property
    def is_ready(self) -> bool:
        """Check if artifacts are loaded and ready for inference."""
        return self.scaler is not None and self.model is not None

    def calculate_risk_level(self, probability: float) -> str:
        """Determine human-readable risk category based on model probability."""
        if probability >= 0.70:
            return "CRITICAL DANGER"
        elif probability >= 0.45:
            return "HIGH HAZARD"
        elif probability >= 0.20:
            return "MODERATE ATTENTION"
        else:
            return "LOW / SAFE"

    def predict_single(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference on a single asteroid's physical features.
        Expected keys in input_data:
        - absolute_magnitude_h
        - estimated_diameter_min_km
        - estimated_diameter_max_km
        - relative_velocity_km_s
        - miss_distance_km
        """
        feature_defaults = {
            "absolute_magnitude_h": 22.0,
            "estimated_diameter_min_km": 0.05,
            "estimated_diameter_max_km": 0.15,
            "estimated_diameter_mean_km": 0.10,
            "relative_velocity_km_s": 15.0,
            "miss_distance_km": 5000000.0,
        }

        clean_input = feature_defaults.copy()
        for k, v in input_data.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                clean_input[k] = float(v)

        if "estimated_diameter_mean_km" not in input_data or input_data.get("estimated_diameter_mean_km") is None:
            d_min = clean_input.get("estimated_diameter_min_km", 0.05)
            d_max = clean_input.get("estimated_diameter_max_km", 0.15)
            clean_input["estimated_diameter_mean_km"] = (d_min + d_max) / 2.0

        if not self.is_ready:
            return {
                "is_hazardous": False,
                "hazard_probability": 0.0,
                "hazard_probability_percent": 0.0,
                "risk_level": "STANDBY",
                "input_features": clean_input,
            }

        df_input = pd.DataFrame([clean_input])[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(df_input)

        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(self.model.predict(X_scaled)[0])
        risk_level = self.calculate_risk_level(proba)

        return {
            "is_hazardous": bool(prediction),
            "hazard_probability": proba,
            "hazard_probability_percent": round(proba * 100, 2),
            "risk_level": risk_level,
            "input_features": clean_input,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a DataFrame of asteroids (e.g. from live API response).
        Adds 'pred_hazardous', 'hazard_probability', and 'risk_level' columns.
        """
        if df.empty:
            df_empty = df.copy()
            df_empty["pred_hazardous"] = 0
            df_empty["hazard_probability"] = 0.0
            df_empty["hazard_probability_pct"] = 0.0
            df_empty["risk_level"] = "STANDBY"
            return df_empty

        df_eval = df.copy()

        # Feature defaults for missing / NaN telemetry fields
        feature_defaults = {
            "absolute_magnitude_h": 22.0,
            "estimated_diameter_min_km": 0.05,
            "estimated_diameter_max_km": 0.15,
            "estimated_diameter_mean_km": 0.10,
            "relative_velocity_km_s": 15.0,
            "miss_distance_km": 5000000.0,
        }

        for col, default_val in feature_defaults.items():
            if col not in df_eval.columns:
                df_eval[col] = default_val

        # Recalculate mean diameter where missing or NaN
        mask_dmean_null = df_eval["estimated_diameter_mean_km"].isna()
        if mask_dmean_null.any():
            dmin_filled = df_eval.loc[mask_dmean_null, "estimated_diameter_min_km"].fillna(0.05)
            dmax_filled = df_eval.loc[mask_dmean_null, "estimated_diameter_max_km"].fillna(0.15)
            df_eval.loc[mask_dmean_null, "estimated_diameter_mean_km"] = (dmin_filled + dmax_filled) / 2.0

        # Fill any remaining NaNs in feature matrix before scaling
        X_feats = df_eval[FEATURE_COLUMNS].fillna(feature_defaults)

        if not self.is_ready:
            preds = df_eval.get("is_potentially_hazardous_asteroid", 0).fillna(0).astype(int).values
            probas = np.where(preds == 1, 0.85, 0.05)
        else:
            try:
                X_scaled = self.scaler.transform(X_feats)
                probas = self.model.predict_proba(X_scaled)[:, 1]
                preds = self.model.predict(X_scaled)
            except Exception as exc:
                logger.error("Error during model inference: %s", exc)
                preds = np.zeros(len(df_eval), dtype=int)
                probas = np.zeros(len(df_eval))

        df_eval["pred_hazardous"] = preds
        df_eval["hazard_probability"] = probas
        df_eval["hazard_probability_pct"] = (probas * 100).round(2)
        df_eval["risk_level"] = [self.calculate_risk_level(p) for p in probas]

        return df_eval
