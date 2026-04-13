import pytest
import pandas as pd
import numpy as np
import yaml
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Test 1: params.yaml has all required keys
# ─────────────────────────────────────────────
def test_params_exist():
    params = load_params()
    assert "base"       in params
    assert "data"       in params
    assert "model"      in params
    assert "features"   in params
    assert "evaluate"   in params
    assert "monitoring" in params


# ─────────────────────────────────────────────
# Test 2: target column is correct
# ─────────────────────────────────────────────
def test_target_column():
    params = load_params()
    assert params["base"]["target"] == "Price"


# ─────────────────────────────────────────────
# Test 3: feature engineering works correctly
# ─────────────────────────────────────────────
def test_feature_engineering():
    from preprocess import feature_engineering

    df = pd.DataFrame({
        "Area"     : [2000, 3000],
        "Bedrooms" : [3, 4],
        "Bathrooms": [2, 3],
        "Floors"   : [2, 1],
        "YearBuilt": [1990, 2005],
        "Location" : ["Downtown", "Rural"],
        "Condition": ["Good", "Fair"],
        "Garage"   : ["Yes", "No"],
        "Price"    : [400000, 500000]
    })

    result = feature_engineering(df.copy())

    # HouseAge should be 2026 - YearBuilt
    assert result["HouseAge"].iloc[0] == 2026 - 1990
    assert result["HouseAge"].iloc[1] == 2026 - 2005

    # TotalRooms = Bedrooms + Bathrooms
    assert result["TotalRooms"].iloc[0] == 3 + 2
    assert result["TotalRooms"].iloc[1] == 4 + 3

    # AreaPerRoom = Area / TotalRooms
    assert result["AreaPerRoom"].iloc[0] == round(2000 / 5, 2)

    # IsNew = 1 if YearBuilt > 2000
    assert result["IsNew"].iloc[0] == 0   # 1990 → old
    assert result["IsNew"].iloc[1] == 1   # 2005 → new


# ─────────────────────────────────────────────
# Test 4: model file exists after training
# ─────────────────────────────────────────────
def test_model_exists():
    assert os.path.exists("model.pkl"), \
        "model.pkl not found — run dvc repro first"


# ─────────────────────────────────────────────
# Test 5: preprocessor file exists
# ─────────────────────────────────────────────
def test_preprocessor_exists():
    assert os.path.exists("data/processed/preprocessor.pkl"), \
        "preprocessor.pkl not found — run dvc repro first"


# ─────────────────────────────────────────────
# Test 6: metrics file exists and has required keys
# ─────────────────────────────────────────────
def test_metrics_exist():
    import json
    assert os.path.exists("data/reports/metrics.json"), \
        "metrics.json not found — run dvc repro first"

    with open("data/reports/metrics.json") as f:
        metrics = json.load(f)

    assert "rmse" in metrics
    assert "mae"  in metrics
    assert "r2"   in metrics
    assert "mape" in metrics


# ─────────────────────────────────────────────
# Test 7: FastAPI input schema validation
# Tests Pydantic model directly without server
# ─────────────────────────────────────────────
def test_fastapi_input_schema():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.main import HouseFeatures
    from pydantic import ValidationError

    # Valid input — should not raise
    valid = HouseFeatures(
        Area=2500, Bedrooms=3, Bathrooms=2,
        Floors=2, YearBuilt=1990,
        Location="Downtown", Condition="Good", Garage="Yes"
    )
    assert valid.Area == 2500
    assert valid.Location == "Downtown"

    # Invalid Location — should raise ValidationError
    with pytest.raises(ValidationError):
        HouseFeatures(
            Area=2500, Bedrooms=3, Bathrooms=2,
            Floors=2, YearBuilt=1990,
            Location="InvalidCity", Condition="Good", Garage="Yes"
        )

    # Invalid Bedrooms (out of range) — should raise ValidationError
    with pytest.raises(ValidationError):
        HouseFeatures(
            Area=2500, Bedrooms=10, Bathrooms=2,
            Floors=2, YearBuilt=1990,
            Location="Downtown", Condition="Good", Garage="Yes"
        )


# ─────────────────────────────────────────────
# Test 8: log transform is reversible
# ─────────────────────────────────────────────
def test_log_transform_reversible():
    prices = [100000, 500000, 999000]
    for p in prices:
        log_p    = np.log1p(p)
        reversed_p = np.expm1(log_p)
        assert abs(reversed_p - p) < 0.01, \
            f"log1p/expm1 not reversible for {p}"