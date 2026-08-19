"""
NeoWatch - Complete ML Training Pipeline Runner
Executes Preprocessing (Checkpoint 2) -> Model Training & Tuning (Checkpoint 3) -> Evaluation Report.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

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
        MODEL_PATH,
        LEGACY_MODEL_PATH,
    )
    from src.data_processor import DataProcessor
    from src.model_trainer import AsteroidModelTrainer
except (ImportError, ModuleNotFoundError):
    from config import (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        LEGACY_RAW_DATA_PATH,
        LEGACY_PROCESSED_DATA_PATH,
        SCALER_PATH,
        MODEL_PATH,
        LEGACY_MODEL_PATH,
    )
    from data_processor import DataProcessor
    from model_trainer import AsteroidModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("NeoWatch.Pipeline")


def run_pipeline():
    print("=" * 70)
    print("NeoWatch - End-to-End Machine Learning Training Pipeline (Plan 2)")
    print("=" * 70)

    raw_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else LEGACY_RAW_DATA_PATH
    if not raw_path.exists():
        print(f"[ERROR] Raw dataset not found at {RAW_DATA_PATH} or {LEGACY_RAW_DATA_PATH}. Please run data collection first.")
        sys.exit(1)

    # -------------------------------------------------------------
    # Step 1: Preprocessing, Scaling & SMOTE (Checkpoint 2)
    # -------------------------------------------------------------
    print("\n>>> Phase 2: Preprocessing, Feature Engineering & SMOTE Balancing...")
    preprocessor = DataProcessor(scaler_type="standard")
    X_train, y_train, X_test, y_test, df_clean = preprocessor.prepare_datasets(
        raw_csv_path=raw_path,
        test_size=0.2,
        random_state=42,
        apply_smote=True,
    )
    print(f"[SUCCESS] Checkpoint 2 Reached:")
    print(f" - Train Samples (after SMOTE): {X_train.shape[0]}")
    print(f" - Test Samples               : {X_test.shape[0]}")
    print(f" - Scaler saved to            : {SCALER_PATH}")
    print(f" - Processed data saved to    : {PROCESSED_DATA_PATH}")

    # -------------------------------------------------------------
    # Step 2: Baseline Model Benchmarking
    # -------------------------------------------------------------
    print("\n>>> Phase 3: Benchmarking Baseline ML Models (5-Fold Cross Validation)...")
    trainer = AsteroidModelTrainer(random_state=42)
    df_benchmarks = trainer.run_benchmarks(X_train, y_train, cv_splits=5)
    print("\n--- Cross-Validation Benchmark Results ---")
    print(df_benchmarks.to_string(index=False))

    # -------------------------------------------------------------
    # Step 3: Hyperparameter Tuning (XGBoost)
    # -------------------------------------------------------------
    print("\n>>> Phase 3: Hyperparameter Optimization for XGBoost (Recall Focused)...")
    best_xgb = trainer.tune_xgboost(X_train, y_train)

    # -------------------------------------------------------------
    # Step 4: Final Test Set Evaluation & Model Serialization (Checkpoint 3)
    # -------------------------------------------------------------
    print("\n>>> Phase 3: Test Set Evaluation & Checkpoint 3 Serialization...")
    eval_results = trainer.evaluate_model(best_xgb, X_test, y_test)
    trainer.save_model(MODEL_PATH)

    print("\n" + "=" * 70)
    print("[SUCCESS] CHECKPOINT 3 COMPLETE: MODEL SUCCESSFULLY TRAINED & SAVED")
    print("=" * 70)
    print(f"Test Recall   : {eval_results['recall']:.4f} (Primary planetary defense metric)")
    print(f"Test ROC-AUC  : {eval_results['roc_auc']:.4f}")
    print(f"Test Precision: {eval_results['precision']:.4f}")
    print(f"Test F1-Score : {eval_results['f1']:.4f}")
    print(f"Model File    : {MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

