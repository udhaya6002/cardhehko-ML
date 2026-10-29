"""
preprocess.py - Train/test split, outlier capping, imputation, encoding, scaling.

Usage:
    from src.preprocess import build_preprocessed_data
    X_train, X_test, y_train, y_test = build_preprocessed_data(df)

Rules (SPEC.md):
    - Split BEFORE any imputation, capping or scaling.
    - Every transformer is fit on training data only.
    - Target is log1p(selling_price).
    - Scaling for linear models only.
    - 1st-99th percentile capping on training data.
    - Median/mode imputation.
    - One-hot encoding.
    - random_state=42.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
CAP_LOWER = 0.01
CAP_UPPER = 0.99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(df: pd.DataFrame, target: str = "selling_price"):
    """80/20 train/test split. Target is log1p-transformed."""
    y = np.log1p(df[target])
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Split: train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def _cap_outliers(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Cap numeric columns at 1st and 99th percentile thresholds
    computed on the training set.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    caps = {}
    for col in numeric_cols:
        lo = X_train[col].quantile(CAP_LOWER)
        hi = X_train[col].quantile(CAP_UPPER)
        caps[col] = (lo, hi)
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col] = X_test[col].clip(lo, hi)

    n_capped = len(caps)
    print(f"  Capped {n_capped} numeric columns at 1st-99th percentile.")
    return X_train, X_test, caps


def _impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Median imputation for numeric columns, mode for categorical.
    Imputers are fit on training data only.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    # Numeric: median
    medians = X_train[numeric_cols].median()
    missing_before = X_train[numeric_cols].isna().sum().sum()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)
    missing_after = X_train[numeric_cols].isna().sum().sum()
    print(f"  Numeric imputation (median): {missing_before} -> {missing_after} missing values.")

    # Categorical: mode
    if len(cat_cols) > 0:
        modes = X_train[cat_cols].mode().iloc[0]
        cat_missing_before = X_train[cat_cols].isna().sum().sum()
        X_train[cat_cols] = X_train[cat_cols].fillna(modes)
        X_test[cat_cols] = X_test[cat_cols].fillna(modes)
        cat_missing_after = X_train[cat_cols].isna().sum().sum()
        print(f"  Categorical imputation (mode): {cat_missing_before} -> {cat_missing_after} missing values.")

    return X_train, X_test


def _encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode categorical columns. Fit on training categories."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return X_train, X_test

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # Align columns: test may have missing or extra dummies
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"  One-hot encoded {len(cat_cols)} categorical columns -> {X_train.shape[1]} features.")
    return X_train, X_test


def _scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standard-scale all numeric columns. Fit on training only."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cols = X_train.columns
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=cols, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_preprocessed_data(df: pd.DataFrame, save: bool = True):
    """
    Full preprocessing pipeline:
        1. Train/test split (80/20) with log1p target
        2. Outlier capping (1st-99th percentile, fit on train)
        3. Median/mode imputation (fit on train)
        4. One-hot encoding (fit on train categories)

    Returns:
        X_train, X_test, y_train, y_test (unscaled - scaling done per model)
        Also saves to data/processed/ if save=True.
    """
    print("Preprocessing pipeline:")

    # 1. Split
    X_train, X_test, y_train, y_test = _split(df)

    # 2. Cap outliers
    X_train, X_test, caps = _cap_outliers(X_train.copy(), X_test.copy())

    # 3. Impute
    X_train, X_test = _impute(X_train, X_test)

    # 4. Encode
    X_train, X_test = _encode(X_train, X_test)

    if save:
        train_df = X_train.copy()
        train_df["selling_price_log"] = y_train
        test_df = X_test.copy()
        test_df["selling_price_log"] = y_test
        train_df.to_csv("data/processed/train.csv", index=False)
        test_df.to_csv("data/processed/test.csv", index=False)
        print(f"  Saved: data/processed/train.csv ({train_df.shape})")
        print(f"  Saved: data/processed/test.csv ({test_df.shape})")

    return X_train, X_test, y_train, y_test


def scale_for_linear(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale data for linear models. Returns scaled copies + scaler."""
    return _scale(X_train.copy(), X_test.copy())
