"""
NeoWatch - Machine Learning Training, Cross-Validation & Optimization Engine
Trains benchmark models (Logistic Regression, Random Forest, LightGBM, XGBoost),
tunes hyperparameters with GridSearchCV, and serializes the best model optimized for Recall and ROC-AUC.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)
import sys
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if "." not in sys.path:
    sys.path.insert(0, ".")

from src.config import MODEL_PATH, LEGACY_MODEL_PATH, FEATURE_COLUMNS

logger = logging.getLogger("NeoWatch.ModelTrainer")


class AsteroidModelTrainer:
    """Trains, optimizes, and evaluates ML models for asteroid hazard detection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=random_state),
            "LightGBM": LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, class_weight="balanced", random_state=random_state, verbose=-1),
            "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, eval_metric="logloss", random_state=random_state),
        }
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.benchmark_results: Dict[str, Dict[str, float]] = {}

    def run_benchmarks(self, X_train: np.ndarray, y_train: np.ndarray, cv_splits: int = 5) -> pd.DataFrame:
        """Run 5-Fold Stratified Cross Validation on all candidate baseline models."""
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        scoring = ["recall", "roc_auc", "f1", "precision"]

        results = []
        for name, model in self.models.items():
            logger.info("Evaluating %s via %d-fold cross-validation...", name, cv_splits)
            cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
            
            res_entry = {
                "Model": name,
                "Recall (Mean)": np.mean(cv_res["test_recall"]),
                "Recall (Std)": np.std(cv_res["test_recall"]),
                "ROC-AUC (Mean)": np.mean(cv_res["test_roc_auc"]),
                "ROC-AUC (Std)": np.std(cv_res["test_roc_auc"]),
                "F1-Score (Mean)": np.mean(cv_res["test_f1"]),
                "Precision (Mean)": np.mean(cv_res["test_precision"]),
            }
            results.append(res_entry)

        df_bench = pd.DataFrame(results).sort_values(by="Recall (Mean)", ascending=False).reset_index(drop=True)
        logger.info("Benchmark summary:\n%s", df_bench.to_string())
        return df_bench

    def tune_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """Tune XGBoost hyperparameters with GridSearchCV optimizing for Recall."""
        logger.info("Starting GridSearchCV for XGBoost hyperparameter tuning...")
        
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "scale_pos_weight": [1.0, 2.0, 3.0],
        }

        xgb = XGBClassifier(eval_metric="logloss", random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring="recall",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        logger.info("Best XGBoost params: %s (Best CV Recall: %.4f)", grid_search.best_params_, grid_search.best_score_)
        self.best_model = grid_search.best_estimator_
        self.best_model_name = "Tuned XGBoost"
        return self.best_model

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive test set evaluation including Confusion Matrix and ROC-AUC."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        eval_report = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=["Safe", "Hazardous"]),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        logger.info("\n--- Model Evaluation Results ---\n%s", eval_report["classification_report"])
        logger.info("Recall: %.4f | ROC-AUC: %.4f | Precision: %.4f | F1: %.4f", recall, auc, precision, f1)
        logger.info("Confusion Matrix:\n%s", cm)
        return eval_report

    def save_model(self, filepath: Optional[Path] = None) -> None:
        """Save best model to models/asteroid_xgb_model.pkl and models/asteroid_model.pkl (Checkpoint 3)."""
        if self.best_model is None:
            raise RuntimeError("No model has been trained or selected yet.")
        target_path = filepath or MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, target_path)
        joblib.dump(self.best_model, LEGACY_MODEL_PATH)
        logger.info("Saved best model (%s) to %s and %s", self.best_model_name, target_path, LEGACY_MODEL_PATH)

    def load_model(self, filepath: Optional[Path] = None) -> Any:
        """Load trained model from disk with fallback support."""
        target_path = filepath or MODEL_PATH
        if not Path(target_path).exists() and Path(LEGACY_MODEL_PATH).exists():
            target_path = LEGACY_MODEL_PATH
        self.best_model = joblib.load(target_path)
        logger.info("Loaded model from %s", target_path)
        return self.best_model
