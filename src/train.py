import pandas as pd
import numpy as np
import yaml
import pickle
import os
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_metrics(y_true, y_pred_log):
    """
    Compute metrics on actual price scale.
    Reverse log1p with expm1 before computing.
    """
    r2_log  = float(r2_score(y_true, y_pred_log))
    mae_log = float(mean_absolute_error(y_true, y_pred_log))

    y_actual = np.expm1(y_true)
    y_pred   = np.expm1(y_pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae  = float(mean_absolute_error(y_actual, y_pred))
    r2   = float(r2_score(y_actual, y_pred))
    mape = float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100)

    return {
        "rmse"   : round(rmse,    2),
        "mae"    : round(mae,     2),
        "r2"     : round(r2,      4),
        "mape"   : round(mape,    2),
        "r2_log" : round(r2_log,  4),
        "mae_log": round(mae_log, 4)
    }


def get_models(params):
    rs = params["base"]["random_state"]
    mp = params["model"]

    return {
        "LinearRegression": (LinearRegression(), {}),

        "Ridge": (
            Ridge(random_state=rs),
            {"alpha": mp["ridge_alpha"]}
        ),

        "Lasso": (
            Lasso(random_state=rs, max_iter=10000),
            {"alpha": mp["lasso_alpha"]}
        ),

        "ElasticNet": (
            ElasticNet(random_state=rs, max_iter=10000),
            {"alpha": mp["elasticnet_alpha"], "l1_ratio": mp["elasticnet_l1_ratio"]}
        ),

        "DecisionTree": (
            DecisionTreeRegressor(
                max_depth        = mp["dt_max_depth"],
                min_samples_split= mp["dt_min_samples_split"],
                random_state     = rs
            ),
            {"max_depth": mp["dt_max_depth"], "min_samples_split": mp["dt_min_samples_split"]}
        ),

        "RandomForest": (
            RandomForestRegressor(
                n_estimators     = mp["n_estimators"],
                max_depth        = mp["max_depth"],
                min_samples_split= mp["min_samples_split"],
                min_samples_leaf = mp["min_samples_leaf"],
                random_state     = rs,
                n_jobs           = -1
            ),
            {
                "n_estimators"     : mp["n_estimators"],
                "max_depth"        : mp["max_depth"],
                "min_samples_split": mp["min_samples_split"],
                "min_samples_leaf" : mp["min_samples_leaf"]
            }
        ),

        "XGBoost": (
            xgb.XGBRegressor(
                n_estimators  = mp["xgb_n_estimators"],
                max_depth     = mp["xgb_max_depth"],
                learning_rate = mp["xgb_learning_rate"],
                subsample     = mp["xgb_subsample"],
                random_state  = rs,
                verbosity     = 0,
                n_jobs        = -1
            ),
            {
                "n_estimators" : mp["xgb_n_estimators"],
                "max_depth"    : mp["xgb_max_depth"],
                "learning_rate": mp["xgb_learning_rate"],
                "subsample"    : mp["xgb_subsample"]
            }
        ),

        "LightGBM": (
            lgb.LGBMRegressor(
                n_estimators  = mp["lgb_n_estimators"],
                max_depth     = mp["lgb_max_depth"],
                learning_rate = mp["lgb_learning_rate"],
                num_leaves    = mp["lgb_num_leaves"],
                random_state  = rs,
                n_jobs        = -1,
                verbose       = -1
            ),
            {
                "n_estimators" : mp["lgb_n_estimators"],
                "max_depth"    : mp["lgb_max_depth"],
                "learning_rate": mp["lgb_learning_rate"],
                "num_leaves"   : mp["lgb_num_leaves"]
            }
        ),
    }


def train():
    params = load_params()
    target = params["base"]["target"]

    train_df = pd.read_csv(params["data"]["processed_train"])
    test_df  = pd.read_csv(params["data"]["processed_test"])

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target]

    models  = get_models(params)
    results = {}
    best_r2    = -np.inf
    best_name  = None
    best_model = None

    print(f"\n{'='*55}")
    print(f"[train] Training {len(models)} models")
    print(f"[train] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"{'='*55}\n")

    for name, (model, model_params) in models.items():
        print(f"[train] Training {name}...")

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)

        if hasattr(model, "feature_importances_"):
            fi = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
            print(f"  Top 3 features : {list(fi.head(3).index)}")

        print(f"  R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:,.0f} | MAPE={metrics['mape']:.1f}%")

        results[name] = metrics

        if metrics["r2"] > best_r2:
            best_r2    = metrics["r2"]
            best_name  = name
            best_model = model

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'R2':>8} {'RMSE':>12} {'MAE':>12} {'MAPE':>8}")
    print(f"{'-'*60}")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["r2"]):
        marker = " <- BEST" if name == best_name else ""
        print(f"{name:<20} {m['r2']:>8.4f} {m['rmse']:>12,.0f} {m['mae']:>12,.0f} {m['mape']:>7.1f}%{marker}")
    print(f"{'='*60}")
    print(f"\n[train] Best model: {best_name} (R2={best_r2:.4f})")

    # Save best model
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save comparison report
    os.makedirs("data/reports", exist_ok=True)
    comparison = {
        "best_model" : best_name,
        "best_r2"    : best_r2,
        "all_results": results
    }
    with open("data/reports/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)

    with open("data/processed/best_model_name.txt", "w") as f:
        f.write(best_name)

    print(f"[train] Best model saved → model.pkl")
    print(f"[train] Comparison saved → data/reports/model_comparison.json")
    print("[train] Done.\n")

    # ── MLFlow will be added in Phase 2 ──


if __name__ == "__main__":
    train()