"""Crop Yield and Production Estimation ML Module.

Independent regression module trained on historical agricultural statistics
for Rice and Maize:
- Predicts estimated yield per unit area (Tonnes / Hectare)
- Calculates estimated total production (Tonnes) based on cultivated area
- Evaluates regression performance using MAE, RMSE, and R²
- Reports prediction uncertainty and regional data limitations

DISCLAIMER:
Yield estimation is a separate macroeconomic/agronomic module driven by regional
climate, soil, and management inputs. It is not inferred from leaf imagery.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "crop_yield_history.csv"
MODEL_PATH = ROOT_DIR / "checkpoints" / "yield_model.pkl"


@dataclass
class YieldPredictionResult:
    crop: str
    state: str
    season: str
    area_hectares: float
    predicted_yield_per_ha: float      # Tonnes / Hectare
    yield_lower_bound: float
    yield_upper_bound: float
    predicted_total_production: float   # Tonnes
    production_lower_bound: float
    production_upper_bound: float
    model_r2_score: float
    model_mae: float
    limitations_note: str


def train_yield_models() -> Dict[str, Any]:
    """Trains regression models on historical records using a leakage-safe split."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing yield dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    features = [
        "state",
        "crop",
        "season",
        "crop_year",
        "area_hectares",
        "annual_rainfall_mm",
        "fertilizer_npk_kg_ha",
        "irrigation_pct",
    ]
    target = "yield_tonnes_per_ha"

    X = df[features]
    y = df[target]

    # Leakage-safe 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    categorical_features = ["state", "crop", "season"]
    numerical_features = [
        "crop_year",
        "area_hectares",
        "annual_rainfall_mm",
        "fertilizer_npk_kg_ha",
        "irrigation_pct",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluate on untouched test set
    y_pred = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "test_mae_tonnes_per_ha": round(mae, 4),
        "test_rmse_tonnes_per_ha": round(rmse, 4),
        "test_r2_score": round(r2, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"pipeline": pipeline, "metrics": metrics}, f)

    print(f"Trained Yield Model saved to {MODEL_PATH}")
    print(f"Test Set Evaluation -> MAE: {mae:.3f} T/ha, RMSE: {rmse:.3f} T/ha, R²: {r2:.3f}")
    return metrics


def get_trained_yield_pipeline():
    """Loads or trains the yield model pipeline."""
    if not MODEL_PATH.exists():
        train_yield_models()

    with MODEL_PATH.open("rb") as f:
        data = pickle.load(f)
    return data["pipeline"], data["metrics"]


def estimate_crop_yield(
    state: str,
    crop: str,
    season: str,
    area_hectares: float,
    annual_rainfall_mm: float,
    fertilizer_npk_kg_ha: float,
    irrigation_pct: float,
    crop_year: int = 2024,
) -> YieldPredictionResult:
    """Estimates crop yield per hectare and total production with uncertainty intervals."""
    pipeline, metrics = get_trained_yield_pipeline()

    input_df = pd.DataFrame(
        [
            {
                "state": state,
                "crop": crop,
                "season": season,
                "crop_year": crop_year,
                "area_hectares": float(area_hectares),
                "annual_rainfall_mm": float(annual_rainfall_mm),
                "fertilizer_npk_kg_ha": float(fertilizer_npk_kg_ha),
                "irrigation_pct": float(irrigation_pct),
            }
        ]
    )

    pred_yield = float(pipeline.predict(input_df)[0])
    mae = metrics["test_mae_tonnes_per_ha"]

    # 90% confidence interval approx +/- 1.64 * MAE
    margin = 1.64 * mae
    yield_lower = max(0.5, pred_yield - margin)
    yield_upper = pred_yield + margin

    prod_est = pred_yield * area_hectares
    prod_lower = yield_lower * area_hectares
    prod_upper = yield_upper * area_hectares

    limitations = (
        f"Prediction based on historical district-level agro-climatic regressions (Model Test R²: {metrics['test_r2_score']:.2f}, "
        f"MAE: {mae:.2f} T/ha). Predictions assume typical pest management and standard seed viability. "
        "Extreme localized microclimatic shocks or severe unmanaged infestations will alter realized field yields."
    )

    return YieldPredictionResult(
        crop=crop,
        state=state,
        season=season,
        area_hectares=area_hectares,
        predicted_yield_per_ha=round(pred_yield, 2),
        yield_lower_bound=round(yield_lower, 2),
        yield_upper_bound=round(yield_upper, 2),
        predicted_total_production=round(prod_est, 1),
        production_lower_bound=round(prod_lower, 1),
        production_upper_bound=round(prod_upper, 1),
        model_r2_score=metrics["test_r2_score"],
        model_mae=mae,
        limitations_note=limitations,
    )


if __name__ == "__main__":
    train_yield_models()
