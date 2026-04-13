import pandas as pd
import numpy as np
import yaml
import pickle
import os
import json
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
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
    Model predicts log1p(Price) → reverse with expm1.
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
            {"alpha"    : mp["elasticnet_alpha"],
             "l1_ratio" : mp["elasticnet_l1_ratio"]}
        ),

        "DecisionTree": (
            DecisionTreeRegressor(
                max_depth        = mp["dt_max_depth"],
                min_samples_split= mp["dt_min_samples_split"],
                random_state     = rs
            ),
            {"max_depth"        : mp["dt_max_depth"],
             "min_samples_split": mp["dt_min_samples_split"]}
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
            {"n_estimators"     : mp["n_estimators"],
             "max_depth"        : mp["max_depth"],
             "min_samples_split": mp["min_samples_split"],
             "min_samples_leaf" : mp["min_samples_leaf"]}
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
            {"n_estimators" : mp["xgb_n_estimators"],
             "max_depth"    : mp["xgb_max_depth"],
             "learning_rate": mp["xgb_learning_rate"],
             "subsample"    : mp["xgb_subsample"]}
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
            {"n_estimators" : mp["lgb_n_estimators"],
             "max_depth"    : mp["lgb_max_depth"],
             "learning_rate": mp["lgb_learning_rate"],
             "num_leaves"   : mp["lgb_num_leaves"]}
        ),
    }


def register_best_model(best_name, best_r2):
    """
    MLFlow Model Registry — 3 stages:
    None → Staging → Production → Archived

    We auto-promote best model to Staging.
    Production promotion is manual (requires human approval).
    This is intentional — production changes should be reviewed.
    """
    client = mlflow.tracking.MlflowClient()

    # Search for the best model run by name
    experiment = mlflow.get_experiment_by_name("house_price_prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{best_name}'",
        order_by=["metrics.r2 DESC"],
        max_results=1
    )

    if not runs:
        print(f"[registry] No run found for {best_name}")
        return

    best_run_id = runs[0].info.run_id
    model_uri   = f"runs:/{best_run_id}/model"

    # Register model in MLFlow registry
    # First time: creates new registered model
    # Subsequent times: creates new version of same model
    try:
        reg = mlflow.register_model(
            model_uri  = model_uri,
            name       = "house_price_model"
        )
        print(f"[registry] Registered model version: {reg.version}")

        # Transition to Staging automatically
        client.transition_model_version_stage(
            name    = "house_price_model",
            version = reg.version,
            stage   = "Staging"
        )
        print(f"[registry] Model v{reg.version} → Staging")
        print(f"[registry] To promote to Production, run:")
        print(f"           mlflow models set-stage house_price_model {reg.version} Production")

    except Exception as e:
        print(f"[registry] Registry error: {e}")


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

    # ── Set MLFlow experiment ──
    # All 8 model runs go under one experiment
    # In MLFlow UI you'll see them all in one place
    mlflow.set_experiment("house_price_prediction")

    for name, (model, model_params) in models.items():
        print(f"[train] Training {name}...")

        # ── Each model = one MLFlow run ──
        with mlflow.start_run(run_name=name) as run:

            # Tag the run so we can search by model name later
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("phase", "training")

            # Log hyperparameters
            # These appear in the "Parameters" tab in MLFlow UI
            mlflow.log_param("model_type",   name)
            mlflow.log_param("train_size",   len(X_train))
            mlflow.log_param("n_features",   X_train.shape[1])
            mlflow.log_param("random_state", params["base"]["random_state"])
            for k, v in model_params.items():
                mlflow.log_param(k, v)

            # Train
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            metrics = get_metrics(y_test, y_pred)

            # Log metrics
            # These appear in the "Metrics" tab in MLFlow UI
            # You can plot them over time across runs
            mlflow.log_metrics(metrics)

            # Log model artifact with input/output signature
            # Signature tells MLFlow what input shape the model expects
            # This is used for model serving and validation
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model      = model,
                artifact_path = "model",
                signature     = signature
            )

            # Log feature importances for tree models
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(
                    model.feature_importances_,
                    index=X_train.columns
                ).sort_values(ascending=False)

                # Save as artifact file
                fi_path = f"data/reports/feature_importance_{name}.json"
                fi.to_json(fi_path)
                mlflow.log_artifact(fi_path)

                print(f"  Top 3 features : {list(fi.head(3).index)}")

            print(f"  Run ID : {run.info.run_id[:8]}...")
            print(f"  R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:,.0f} | MAPE={metrics['mape']:.1f}%")

        results[name] = metrics

        if metrics["r2"] > best_r2:
            best_r2    = metrics["r2"]
            best_name  = name
            best_model = model

    # ── Comparison table ──
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'R2':>8} {'RMSE':>12} {'MAE':>12} {'MAPE':>8}")
    print(f"{'-'*60}")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["r2"]):
        marker = " <- BEST" if name == best_name else ""
        print(f"{name:<20} {m['r2']:>8.4f} {m['rmse']:>12,.0f} {m['mae']:>12,.0f} {m['mape']:>7.1f}%{marker}")
    print(f"{'='*60}")
    print(f"\n[train] Best model: {best_name} (R2={best_r2:.4f})")

    # ── Save best model locally ──
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # ── Register best model in MLFlow registry ──
    print(f"\n[train] Registering best model in MLFlow registry...")
    register_best_model(best_name, best_r2)

    # ── Save comparison report ──
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

    print(f"[train] Best model saved    → model.pkl")
    print(f"[train] Comparison saved    → data/reports/model_comparison.json")
    print(f"[train] MLFlow UI           → run: mlflow ui")
    print("[train] Done.\n")


if __name__ == "__main__":
    train()