"""
train_linear.py - Linear Regression (baseline), Ridge, and Lasso with GridSearchCV.

Usage:
    python -m src.train_linear

Rules (SPEC.md):
    - 5-fold KFold, shuffle=True, random_state=42, training set only.
    - Scaling for linear models only.
    - Report R², RMSE, MAE on both log and rupee scale.
    - Save every decision to reports/decisions.md.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, make_scorer
)

from src.preprocess import scale_for_linear

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def _metrics_log(y_true, y_pred):
    """Metrics on the log1p scale."""
    return {
        "r2_log": float(r2_score(y_true, y_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
    }


def _metrics_rupee(y_true_log, y_pred_log):
    """Metrics on the rupee (expm1) scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "r2_rupee": float(r2_score(y_true, y_pred)),
        "rmse_rupee": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_rupee": float(mean_absolute_error(y_true, y_pred)),
    }


def _evaluate(model, X_test, y_test, name):
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)
    log_m = _metrics_log(y_test, y_pred)
    rupee_m = _metrics_rupee(y_test, y_pred)
    print(f"\n  {name} Test Results:")
    print(f"    Log   - R²: {log_m['r2_log']:.4f}, RMSE: {log_m['rmse_log']:.4f}, MAE: {log_m['mae_log']:.4f}")
    print(f"    Rupee - R²: {rupee_m['r2_rupee']:.4f}, RMSE: Rs.{rupee_m['rmse_rupee']:,.0f}, MAE: Rs.{rupee_m['mae_rupee']:,.0f}")
    return {**log_m, **rupee_m}


def train_linear_models(X_train, X_test, y_train, y_test):
    """
    Train baseline Linear Regression, Ridge, and Lasso.
    Returns a list of result dicts.
    """
    # Scale for linear models
    X_train_s, X_test_s, scaler = scale_for_linear(X_train, X_test)

    results = []

    # --- 1. Baseline Linear Regression ---
    print("\n[1/3] Linear Regression (baseline)...")
    lr = LinearRegression()
    cv_scores = cross_validate(lr, X_train_s, y_train, cv=CV,
                               scoring=["r2", "neg_mean_squared_error"],
                               return_train_score=False)
    lr.fit(X_train_s, y_train)
    cv_r2 = cv_scores["test_r2"].mean()
    cv_rmse = np.sqrt(-cv_scores["test_neg_mean_squared_error"].mean())
    print(f"  CV R²: {cv_r2:.4f}, CV RMSE: {cv_rmse:.4f}")
    test_m = _evaluate(lr, X_test_s, y_test, "Linear Regression")
    results.append({
        "model": "Linear Regression",
        "best_params": "N/A",
        "cv_r2": float(cv_r2),
        **test_m,
    })
    joblib.dump({"model": lr, "scaler": scaler}, "models/linear_regression.joblib")

    # --- 2. Ridge ---
    print("\n[2/3] Ridge (GridSearchCV over alpha)...")
    alphas = {"alpha": [0.01, 0.1, 1, 10, 100, 1000]}
    ridge_gs = GridSearchCV(Ridge(random_state=RANDOM_STATE), alphas, cv=CV,
                            scoring="r2", refit=True, n_jobs=-1)
    ridge_gs.fit(X_train_s, y_train)
    print(f"  Best alpha: {ridge_gs.best_params_['alpha']}, CV R²: {ridge_gs.best_score_:.4f}")
    test_m = _evaluate(ridge_gs.best_estimator_, X_test_s, y_test, "Ridge")
    results.append({
        "model": "Ridge",
        "best_params": str(ridge_gs.best_params_),
        "cv_r2": float(ridge_gs.best_score_),
        **test_m,
    })
    joblib.dump({"model": ridge_gs.best_estimator_, "scaler": scaler}, "models/ridge.joblib")

    # --- 3. Lasso ---
    print("\n[3/3] Lasso (GridSearchCV over alpha)...")
    alphas_lasso = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    lasso_gs = GridSearchCV(Lasso(random_state=RANDOM_STATE, max_iter=10000),
                            alphas_lasso, cv=CV, scoring="r2", refit=True, n_jobs=-1)
    lasso_gs.fit(X_train_s, y_train)
    print(f"  Best alpha: {lasso_gs.best_params_['alpha']}, CV R²: {lasso_gs.best_score_:.4f}")
    test_m = _evaluate(lasso_gs.best_estimator_, X_test_s, y_test, "Lasso")
    results.append({
        "model": "Lasso",
        "best_params": str(lasso_gs.best_params_),
        "cv_r2": float(lasso_gs.best_score_),
        **test_m,
    })
    joblib.dump({"model": lasso_gs.best_estimator_, "scaler": scaler}, "models/lasso.joblib")

    # Save intermediate results
    with open("results/ridge_lasso.json", "w") as f:
        json.dump(results[1:], f, indent=2)

    return results
