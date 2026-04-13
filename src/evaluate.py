import pandas as pd
import yaml
import pickle
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate():
    params = load_params()
    target = params["base"]["target"]

    test_df = pd.read_csv(params["data"]["processed_test"])

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]  # this is log1p(Price)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred_log = model.predict(X_test)

    # ── Metrics on log scale (what model optimises) ──
    r2_log  = float(r2_score(y_test, y_pred_log))
    mae_log = float(mean_absolute_error(y_test, y_pred_log))

    # ── Reverse log transform → actual price ──
    # Model predicts log1p(Price), we applied log1p in preprocess.py
    # expm1 is the exact inverse: expm1(log1p(x)) = x
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    # ── Metrics on actual price scale (meaningful to humans) ──
    rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)))
    mae  = float(mean_absolute_error(y_test_actual, y_pred_actual))
    r2   = float(r2_score(y_test_actual, y_pred_actual))
    mape = float(np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100)

    metrics = {
        "rmse"   : round(rmse,    2),
        "mae"    : round(mae,     2),
        "r2"     : round(r2,      4),
        "mape"   : round(mape,    2),
        "r2_log" : round(r2_log,  4),
        "mae_log": round(mae_log, 4)
    }

    print(f"\n{'='*45}")
    print(f"[evaluate] ── Actual Price Scale ──")
    print(f"  RMSE : {metrics['rmse']:>12,.2f}")
    print(f"  MAE  : {metrics['mae']:>12,.2f}")
    print(f"  R2   : {metrics['r2']:>12.4f}")
    print(f"  MAPE : {metrics['mape']:>11.2f}%")
    print(f"[evaluate] ── Log Scale ──")
    print(f"  R2   : {metrics['r2_log']:>12.4f}")
    print(f"  MAE  : {metrics['mae_log']:>12.4f}")
    print(f"{'='*45}")

    if metrics["r2"] > 0.85:
        print("[evaluate] Model quality: GOOD")
    elif metrics["r2"] > 0.70:
        print("[evaluate] Model quality: ACCEPTABLE")
    else:
        print("[evaluate] Model quality: POOR (expected for synthetic data)")

    os.makedirs("data/reports", exist_ok=True)
    with open(params["evaluate"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n[evaluate] Metrics saved to {params['evaluate']['metrics_path']}")
    print("[evaluate] Done.")


if __name__ == "__main__":
    evaluate()