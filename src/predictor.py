"""
NeoWatch - Real-Time Inference & Risk Scoring Engine
Loads serialized model and scaler artifacts to generate real-time hazard predictions and risk scores.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import joblib

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
        if Path(self.scaler_path).exists():
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Loaded scaler from %s", self.scaler_path)
        else:
            logger.warning("Scaler artifact not found at %s", self.scaler_path)

        if Path(self.model_path).exists():
            self.model = joblib.load(self.model_path)
            logger.info("Loaded model from %s", self.model_path)
        elif Path(LEGACY_MODEL_PATH).exists():
            self.model = joblib.load(LEGACY_MODEL_PATH)
            logger.info("Loaded fallback model from %s", LEGACY_MODEL_PATH)
        else:
            logger.warning("Model artifact not found at %s or %s", self.model_path, LEGACY_MODEL_PATH)

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
        if not self.is_ready:
            raise RuntimeError("Predictor artifacts (scaler or model) are not loaded.")

        # Compute mean diameter if missing
        if "estimated_diameter_mean_km" not in input_data:
            d_min = input_data.get("estimated_diameter_min_km", 0.0)
            d_max = input_data.get("estimated_diameter_max_km", d_min)
            input_data["estimated_diameter_mean_km"] = (d_min + d_max) / 2.0

        df_input = pd.DataFrame([input_data])[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(df_input)

        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(self.model.predict(X_scaled)[0])
        risk_level = self.calculate_risk_level(proba)

        return {
            "is_hazardous": bool(prediction),
            "hazard_probability": proba,
            "hazard_probability_percent": round(proba * 100, 2),
            "risk_level": risk_level,
            "input_features": input_data,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a DataFrame of asteroids (e.g. from live API response).
        Adds 'pred_hazardous', 'hazard_probability', and 'risk_level' columns.
        """
        if not self.is_ready:
            raise RuntimeError("Predictor artifacts (scaler or model) are not loaded.")

        df_eval = df.copy()

        # Ensure estimated_diameter_mean_km exists
        if "estimated_diameter_mean_km" not in df_eval.columns:
            df_eval["estimated_diameter_mean_km"] = (
                df_eval["estimated_diameter_min_km"] + df_eval["estimated_diameter_max_km"]
            ) / 2.0

        X_feats = df_eval[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X_feats)

        probas = self.model.predict_proba(X_scaled)[:, 1]
        preds = self.model.predict(X_scaled)

        df_eval["pred_hazardous"] = preds
        df_eval["hazard_probability"] = probas
        df_eval["hazard_probability_pct"] = (probas * 100).round(2)
        df_eval["risk_level"] = [self.calculate_risk_level(p) for p in probas]

        return df_eval
