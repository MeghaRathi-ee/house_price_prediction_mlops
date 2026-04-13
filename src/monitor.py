import pandas as pd
import numpy as np
import yaml
import json
import os
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric
)


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# WHAT IS DATA DRIFT?
# ─────────────────────────────────────────────
# When you deploy a model, it was trained on
# data from a specific time period.
# Over time, the real world changes:
# - House prices go up
# - New neighborhoods become popular
# - Building trends change
#
# When incoming data looks different from
# training data → model predictions degrade
# This is called DATA DRIFT
#
# Evidently detects this BEFORE performance
# drops — so you can retrain proactively
# ─────────────────────────────────────────────


def simulate_new_data(reference_df, target, n=200):
    """
    In production, new_data = real incoming requests
    to your FastAPI endpoint logged to a database.

    Here we simulate drift by:
    - Shifting Area distribution upward (houses getting bigger)
    - Shifting YearBuilt toward recent years
    - Adding slight noise to other features

    This simulates what would happen 2-3 years after deployment.
    """
    new_df = reference_df.sample(n=n, replace=True).copy()

    # Simulate drift — Area increases over time
    new_df["Area"] = new_df["Area"] * np.random.uniform(1.1, 1.4, size=n)
    new_df["Area"] = new_df["Area"].astype(int)

    # Simulate drift — newer houses being built
    new_df["YearBuilt"] = new_df["YearBuilt"] + np.random.randint(5, 20, size=n)
    new_df["YearBuilt"] = new_df["YearBuilt"].clip(upper=2024)

    # Simulate slight price drift
    new_df[target] = new_df[target] * np.random.uniform(1.05, 1.3, size=n)

    print(f"[simulate] Reference Area mean : {reference_df['Area'].mean():.0f}")
    print(f"[simulate] New data  Area mean : {new_df['Area'].mean():.0f}")
    print(f"[simulate] Reference YearBuilt mean : {reference_df['YearBuilt'].mean():.0f}")
    print(f"[simulate] New data  YearBuilt mean : {new_df['YearBuilt'].mean():.0f}")

    return new_df


def run_drift_report(reference_df, current_df, params):
    """
    Evidently compares reference (training) data
    vs current (new incoming) data.

    It runs statistical tests per column:
    - Numerical: Kolmogorov-Smirnov test
    - Categorical: Chi-squared test

    If p-value < threshold → drift detected
    """
    print("\n[monitor] Running Evidently drift report...")

    # ── Full drift report ──
    # DataDriftPreset: checks every feature for drift
    # DataQualityPreset: checks missing values, ranges, etc
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])

    report.run(
        reference_data = reference_df,
        current_data   = current_df
    )

    # Save HTML report — visual dashboard
    os.makedirs("data/reports", exist_ok=True)
    report_path = params["monitoring"]["report_path"]
    report.save_html(report_path)
    print(f"[monitor] HTML report saved → {report_path}")

    # Save JSON report — for programmatic use
    json_path = report_path.replace(".html", ".json")
    report.save_json(json_path)

    return report, json_path


def parse_drift_results(json_path, params):
    """
    Parse Evidently JSON output to extract:
    - Which features drifted
    - Overall dataset drift score
    - Whether retraining should be triggered
    """
    with open(json_path) as f:
        report_data = json.load(f)

    threshold    = params["monitoring"]["drift_threshold"]
    drifted_cols = []
    drift_scores = {}

    # Walk through metrics in report
    for metric in report_data.get("metrics", []):
        metric_id = metric.get("metric", "")
        result    = metric.get("result", {})

        # DatasetDriftMetric gives overall drift
        if "DatasetDriftMetric" in metric_id:
            share_drifted = result.get("share_of_drifted_columns", 0)
            n_drifted     = result.get("number_of_drifted_columns", 0)
            dataset_drift = result.get("dataset_drift", False)

            print(f"\n[monitor] ── Drift Summary ──")
            print(f"  Drifted columns    : {n_drifted}")
            print(f"  Share drifted      : {share_drifted:.2%}")
            print(f"  Dataset drift flag : {dataset_drift}")

        # ColumnDriftMetric gives per-column drift
        if "ColumnDriftMetric" in metric_id:
            col      = result.get("column_name", "")
            drifted  = result.get("drift_detected", False)
            p_value  = result.get("p_value", 1.0)
            drift_scores[col] = p_value

            if drifted:
                drifted_cols.append(col)
                print(f"  DRIFT: {col:<20} p={p_value:.4f}")

    return drifted_cols, drift_scores, share_drifted if 'share_drifted' in dir() else 0


def check_retrain_trigger(drifted_cols, share_drifted, params):
    """
    Retraining trigger logic.

    We trigger retraining if:
    1. Share of drifted columns > threshold (default 0.05 = 5%)
    2. OR any critical feature drifted (Area, Price)

    In production this would:
    - Send a Slack/email alert
    - Trigger an Airflow DAG
    - Create a GitHub issue
    """
    threshold       = params["monitoring"]["drift_threshold"]
    critical_cols   = ["Area", "YearBuilt", "Price"]
    critical_drift  = any(col in drifted_cols for col in critical_cols)
    threshold_breach= share_drifted > threshold

    print(f"\n[monitor] ── Retraining Trigger Check ──")
    print(f"  Drift threshold    : {threshold}")
    print(f"  Share drifted      : {share_drifted:.2%}")
    print(f"  Threshold breached : {threshold_breach}")
    print(f"  Critical col drift : {critical_drift}")
    print(f"  Drifted cols       : {drifted_cols}")

    should_retrain = threshold_breach or critical_drift

    trigger_result = {
        "timestamp"        : datetime.now().isoformat(),
        "should_retrain"   : should_retrain,
        "reason"           : [],
        "drifted_columns"  : drifted_cols,
        "share_drifted"    : share_drifted,
        "threshold"        : threshold
    }

    if threshold_breach:
        trigger_result["reason"].append(
            f"Share of drifted columns {share_drifted:.2%} > threshold {threshold}"
        )
    if critical_drift:
        trigger_result["reason"].append(
            f"Critical features drifted: {[c for c in critical_cols if c in drifted_cols]}"
        )

    if should_retrain:
        print(f"\n  ⚠️  RETRAIN TRIGGER FIRED")
        for r in trigger_result["reason"]:
            print(f"     Reason: {r}")
        print(f"     Action: Run 'dvc repro' to retrain pipeline")
    else:
        print(f"\n  ✅  No retraining needed")

    # Save trigger result
    trigger_path = "data/reports/retrain_trigger.json"
    with open(trigger_path, "w") as f:
        json.dump(trigger_result, f, indent=4)
    print(f"\n[monitor] Trigger result saved → {trigger_path}")

    return should_retrain


def monitor():
    params = load_params()
    target = params["base"]["target"]

    print(f"\n{'='*55}")
    print(f"[monitor] Starting monitoring pipeline")
    print(f"{'='*55}\n")

    # Load reference data (training data)
    # This is the baseline — what model was trained on
    reference_df = pd.read_csv(params["data"]["processed_train"])
    print(f"[monitor] Reference data shape: {reference_df.shape}")

    # In production: load from database/API logs
    # Here: simulate drift for demonstration
    print(f"\n[monitor] Simulating new incoming data with drift...")
    current_df = simulate_new_data(reference_df, target, n=200)

    # Run Evidently report
    report, json_path = run_drift_report(reference_df, current_df, params)

    # Parse results
    drifted_cols, drift_scores, share_drifted = parse_drift_results(json_path, params)

    # Check if retraining should be triggered
    should_retrain = check_retrain_trigger(drifted_cols, share_drifted, params)

    print(f"\n{'='*55}")
    print(f"[monitor] Done.")
    print(f"[monitor] Open data/reports/drift_report.html to see visual report")
    print(f"{'='*55}\n")

    return should_retrain


if __name__ == "__main__":
    monitor()