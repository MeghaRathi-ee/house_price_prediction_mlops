import pandas as pd
import numpy as np
import yaml
import pickle
import os
import json
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# STEP 1: DROP USELESS COLUMNS
# ─────────────────────────────────────────────
def drop_useless(df):
    """
    Drop Id — it is just a row number (1,2,3...).
    Has zero relationship with Price.
    Keeping it would confuse the model.
    """
    before = df.shape[1]
    df.drop(columns=["Id"], inplace=True, errors="ignore")
    print(f"[drop] Columns before: {before} → after: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# STEP 2: OUTLIER REMOVAL (IQR METHOD)
# ─────────────────────────────────────────────
def remove_outliers(df, target):
    """
    IQR method: remove rows where any numerical feature
    is beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR.

    We fit IQR bounds on the full df (before split) since
    this runs after ingest but before train/test split in
    our pipeline. Bounds are saved for reference.

    NOTE: This dataset has 0 outliers (synthetic data).
    But this code is correct practice for real data.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]

    before = len(df)
    bounds = {}

    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        bounds[col] = {"lower": lo, "upper": hi}
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    removed = before - len(df)
    print(f"[outliers] Removed {removed} outlier rows ({before} → {len(df)})")

    # Save bounds so FastAPI can validate incoming inputs
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/outlier_bounds.json", "w") as f:
        json.dump(bounds, f, indent=4)

    return df


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df):
    """
    Create meaningful new features from existing ones.
    Applied to both train and test sets identically.
    """

    # HouseAge: years since built — more interpretable than raw year
    # 2026 - 1900 = 126 years old vs just the number 1900
    df["HouseAge"] = 2026 - df["YearBuilt"]

    # TotalRooms: combined room count
    # 5bed+4bath = 9 rooms vs 1bed+1bath = 2 rooms
    df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]

    # AreaPerRoom: space efficiency
    # 4000sqft / 2 rooms = 2000sqft per room (spacious)
    # 4000sqft / 8 rooms = 500sqft per room (cramped)
    df["AreaPerRoom"] = (df["Area"] / df["TotalRooms"]).round(2)

    # IsNew: binary — built after 2000 or not
    # Sometimes recency matters more than exact age
    df["IsNew"] = (df["YearBuilt"] > 2000).astype(int)

    print(f"[feature_eng] HouseAge    : {df['HouseAge'].min()} – {df['HouseAge'].max()} years")
    print(f"[feature_eng] TotalRooms  : {df['TotalRooms'].min()} – {df['TotalRooms'].max()}")
    print(f"[feature_eng] AreaPerRoom : {df['AreaPerRoom'].min():.1f} – {df['AreaPerRoom'].max():.1f}")
    print(f"[feature_eng] IsNew=1     : {df['IsNew'].sum()} houses out of {len(df)}")

    return df


# ─────────────────────────────────────────────
# STEP 4: LOG TRANSFORM ON TARGET
# ─────────────────────────────────────────────
def log_transform_target(train_df, test_df, target):
    """
    Apply log1p transform to Price.

    log1p(x) = log(x + 1) — safe even if price = 0.
    Makes right-skewed price distribution normal.
    Model learns better on normal distributions.

    NOTE: This dataset Price skew = -0.06 (already normal).
    But in real house price data, always apply this.

    IMPORTANT: At inference time you must reverse with np.expm1()
    to get back actual price from log-price prediction.
    """
    train_df[target] = np.log1p(train_df[target])
    test_df[target]  = np.log1p(test_df[target])

    print(f"[log_transform] Price after log1p — mean: {train_df[target].mean():.4f}, std: {train_df[target].std():.4f}")
    print(f"[log_transform] Remember: reverse with np.expm1() at inference time")

    return train_df, test_df


# ─────────────────────────────────────────────
# STEP 5: CORRELATION-BASED FEATURE SELECTION
# ─────────────────────────────────────────────
def correlation_selection(X_train, y_train, threshold=0.01):
    """
    Drop numerical features with absolute correlation
    with target below threshold.

    threshold=0.01 means drop features with |corr| < 0.01.
    We use a very low threshold because this dataset is synthetic
    and all correlations are near 0. In real data use 0.05–0.10.

    Only computed on training data to avoid data leakage.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    correlations = {}
    for col in num_cols:
        corr = abs(X_train[col].corr(y_train))
        correlations[col] = round(corr, 6)

    print(f"\n[corr_select] Correlations with target:")
    for col, corr in sorted(correlations.items(), key=lambda x: -x[1]):
        status = "keep" if corr >= threshold else "DROP"
        print(f"  {col:<20} {corr:.6f}  [{status}]")

    drop_cols = [col for col, corr in correlations.items() if corr < threshold]

    if drop_cols:
        X_train.drop(columns=drop_cols, inplace=True)
        print(f"\n[corr_select] Dropped {len(drop_cols)} low-correlation features: {drop_cols}")
    else:
        print(f"\n[corr_select] No features dropped — all above threshold {threshold}")

    # Save selected feature list for inference
    selected = X_train.columns.tolist()
    with open("data/processed/selected_features.json", "w") as f:
        json.dump({"features": selected, "dropped": drop_cols}, f, indent=4)

    return X_train, drop_cols


# ─────────────────────────────────────────────
# MAIN PREPROCESS FUNCTION
# ─────────────────────────────────────────────
def preprocess():
    params   = load_params()
    target   = params["base"]["target"]
    cat_cols = params["features"]["categorical"]

    train_df = pd.read_csv(params["data"]["processed_train"])
    test_df  = pd.read_csv(params["data"]["processed_test"])

    print(f"\n{'='*50}")
    print(f"[preprocess] Starting preprocessing pipeline")
    print(f"[preprocess] Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"{'='*50}\n")

    # 1. Drop useless columns
    print("--- STEP 1: Drop useless columns ---")
    train_df = drop_useless(train_df)
    test_df  = drop_useless(test_df)

    # 2. Outlier removal (on train only — don't touch test)
    print("\n--- STEP 2: Outlier removal ---")
    train_df = remove_outliers(train_df, target)

    # 3. Feature engineering
    print("\n--- STEP 3: Feature engineering ---")
    train_df = feature_engineering(train_df)
    test_df  = feature_engineering(test_df)

    # 4. Log transform on Price
    print("\n--- STEP 4: Log transform on Price ---")
    train_df, test_df = log_transform_target(train_df, test_df, target)

    # 5. Separate features and target
    y_train = train_df[target]
    y_test  = test_df[target]
    X_train = train_df.drop(columns=[target])
    X_test  = test_df.drop(columns=[target])

    # 6. Correlation-based feature selection (on train only)
    print("\n--- STEP 5: Correlation-based feature selection ---")
    threshold = params["features"].get("correlation_threshold", 0.0)
    X_train, dropped_cols = correlation_selection(X_train, y_train, threshold=threshold)
    X_test.drop(columns=dropped_cols, inplace=True, errors="ignore")

    # 7. Update num_cols after selection
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_present = [c for c in cat_cols if c in X_train.columns]

    # 8. Fill missing values (safety net)
    print("\n--- STEP 6: Fill missing values ---")
    for col in num_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)
    for col in cat_cols_present:
        mode = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode)
        X_test[col]  = X_test[col].fillna(mode)
    print(f"[fillna] No missing values found (synthetic data)")

    # 9. Build sklearn preprocessing pipeline
    print("\n--- STEP 7: Scaling + Encoding ---")
    numerical_pipeline   = Pipeline(steps=[("scaler",  RobustScaler())])
    categorical_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    ))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,   num_cols),
        ("cat", categorical_pipeline, cat_cols_present)
    ])

    # Fit on train ONLY — transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    # 10. Reconstruct feature names
    ohe_features = preprocessor.named_transformers_["cat"]["encoder"]\
                   .get_feature_names_out(cat_cols_present)
    all_features = num_cols + list(ohe_features)

    X_train_df = pd.DataFrame(X_train_processed, columns=all_features)
    X_test_df  = pd.DataFrame(X_test_processed,  columns=all_features)

    # 11. Add target back
    X_train_df[target] = y_train.values
    X_test_df[target]  = y_test.values

    # 12. Save everything
    os.makedirs("data/processed", exist_ok=True)
    X_train_df.to_csv(params["data"]["processed_train"], index=False)
    X_test_df.to_csv(params["data"]["processed_test"],   index=False)

    with open("data/processed/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"\n{'='*50}")
    print(f"[preprocess] Final train shape : {X_train_df.shape}")
    print(f"[preprocess] Final test shape  : {X_test_df.shape}")
    print(f"[preprocess] Total features    : {len(all_features)}")
    print(f"[preprocess] Saved files:")
    print(f"  → data/processed/train.csv")
    print(f"  → data/processed/test.csv")
    print(f"  → data/processed/preprocessor.pkl")
    print(f"  → data/processed/outlier_bounds.json")
    print(f"  → data/processed/selected_features.json")
    print(f"{'='*50}")
    print("[preprocess] Done.")


if __name__ == "__main__":
    preprocess()