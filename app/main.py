import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "House Price Prediction API",
    description = "Predicts house price given features. Built with FastAPI + MLOps pipeline.",
    version     = "1.0.0"
)


# ─────────────────────────────────────────────
# Load artifacts once at startup
# Not inside predict function — loading model
# on every request would be very slow
# ─────────────────────────────────────────────
model            = None
preprocessor     = None
outlier_bounds   = {}
selected_features= None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, outlier_bounds, selected_features

    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        print("[startup] model.pkl loaded")
    except FileNotFoundError:
        raise RuntimeError("model.pkl not found. Run dvc repro first.")

    try:
        with open("data/processed/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        print("[startup] preprocessor.pkl loaded")
    except FileNotFoundError:
        raise RuntimeError("preprocessor.pkl not found. Run dvc repro first.")

    try:
        with open("data/processed/outlier_bounds.json") as f:
            outlier_bounds = json.load(f)
        print("[startup] outlier_bounds.json loaded")
    except FileNotFoundError:
        outlier_bounds = {}

    try:
        with open("data/processed/selected_features.json") as f:
            selected_features = json.load(f)["features"]
        print("[startup] selected_features.json loaded")
    except FileNotFoundError:
        selected_features = None

    print("[startup] All artifacts loaded. API ready.")


# ─────────────────────────────────────────────
# Input schema — Pydantic validates every field
# Wrong type or missing field = 422 error auto
# ─────────────────────────────────────────────
class HouseFeatures(BaseModel):
    Area      : int = Field(..., gt=0,            description="Area in sqft")
    Bedrooms  : int = Field(..., ge=1, le=5,      description="Bedrooms (1-5)")
    Bathrooms : int = Field(..., ge=1, le=4,      description="Bathrooms (1-4)")
    Floors    : int = Field(..., ge=1, le=3,      description="Floors (1-3)")
    YearBuilt : int = Field(..., ge=1900, le=2024,description="Year built")
    Location  : Literal["Downtown","Suburban","Urban","Rural"]
    Condition : Literal["Excellent","Good","Fair","Poor"]
    Garage    : Literal["Yes","No"]

    class Config:
        json_schema_extra = {
            "example": {
                "Area"     : 2500,
                "Bedrooms" : 3,
                "Bathrooms": 2,
                "Floors"   : 2,
                "YearBuilt": 1990,
                "Location" : "Downtown",
                "Condition": "Good",
                "Garage"   : "Yes"
            }
        }


# ─────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_price    : float
    predicted_price_fmt: str
    model_used         : str
    features_used      : dict


# ─────────────────────────────────────────────
# Feature engineering
# MUST match preprocess.py exactly
# If you change preprocess.py, update this too
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["HouseAge"]    = 2026 - df["YearBuilt"]
    df["TotalRooms"]  = df["Bedrooms"] + df["Bathrooms"]
    df["AreaPerRoom"] = (df["Area"] / df["TotalRooms"]).round(2)
    df["IsNew"]       = (df["YearBuilt"] > 2000).astype(int)
    return df


# ─────────────────────────────────────────────
# Validate input against outlier bounds
# ─────────────────────────────────────────────
def validate_input(data: dict) -> list:
    warnings = []
    for col, bounds in outlier_bounds.items():
        if col in data:
            val = data[col]
            if val < bounds["lower"] or val > bounds["upper"]:
                warnings.append(
                    f"{col}={val} outside training range "
                    f"[{bounds['lower']:.1f}, {bounds['upper']:.1f}]"
                )
    return warnings


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "docs"   : "/docs",
        "health" : "/health",
        "predict": "POST /predict"
    }


@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "preprocessor": preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """
    Predict house price.
    Steps:
    1. Convert input to DataFrame
    2. Validate against outlier bounds
    3. Apply feature engineering (same as preprocess.py)
    4. Transform using fitted preprocessor
    5. Predict log1p(Price)
    6. Reverse with expm1 → actual price
    7. Return formatted response
    """
    data = features.dict()
    df   = pd.DataFrame([data])

    # Validate
    input_warnings = validate_input(data)

    # Feature engineering
    df = engineer_features(df)

    # Drop Id if present
    df.drop(columns=["Id"], inplace=True, errors="ignore")

    try:
        # Preprocess
        X_processed  = preprocessor.transform(df)

        # Reconstruct feature names so sklearn does not warn
        cat_cols_api = ["Location", "Condition", "Garage"]
        ohe_features = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols_api)
        num_cols_api = [c for c in df.columns if c not in cat_cols_api]
        all_features = list(num_cols_api) + list(ohe_features)
        X_df         = pd.DataFrame(X_processed, columns=all_features)

        # Predict log price
        log_pred = model.predict(X_df)[0]







        # Reverse log transform
        predicted_price = float(np.expm1(log_pred))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        predicted_price     = round(predicted_price, 2),
        predicted_price_fmt = f"${predicted_price:,.0f}",
        model_used          = type(model).__name__,
        features_used       = {
            **data,
            "HouseAge"   : int(2026 - data["YearBuilt"]),
            "TotalRooms" : int(data["Bedrooms"] + data["Bathrooms"]),
            "AreaPerRoom": round(data["Area"] / (data["Bedrooms"] + data["Bathrooms"]), 2),
            "IsNew"      : int(data["YearBuilt"] > 2000),
            "warnings"   : input_warnings
        }
    )


@app.get("/model-info")
def model_info():
    return {
        "model_type"  : type(model).__name__,
        "model_params": model.get_params() if hasattr(model, "get_params") else {},
    }