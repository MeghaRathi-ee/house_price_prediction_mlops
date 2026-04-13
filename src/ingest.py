import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def ingest():
    params = load_params()

    raw_path   = params["data"]["raw_path"]
    test_size  = params["data"]["test_size"]
    random_state = params["base"]["random_state"]
    target     = params["base"]["target"]
    drop_cols  = params["features"]["drop_cols"]

    print(f"[ingest] Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)

    print(f"[ingest] Raw dataset shape: {df.shape}")
    print(f"[ingest] Columns: {list(df.columns)}")

    # Drop useless columns (Id is just a row number, not a feature)
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    print(f"[ingest] Dropped columns: {drop_cols}")

    # Check target exists
    assert target in df.columns, f"Target column '{target}' not found!"

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    print(f"[ingest] Target stats — min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Reconstruct full train/test DataFrames
    train_df = X_train.copy()
    train_df[target] = y_train.values

    test_df = X_test.copy()
    test_df[target] = y_test.values

    # Save
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(params["data"]["processed_train"], index=False)
    test_df.to_csv(params["data"]["processed_test"], index=False)

    print(f"[ingest] Train size: {train_df.shape}")
    print(f"[ingest] Test size : {test_df.shape}")
    print("[ingest] Done.")


if __name__ == "__main__":
    ingest()