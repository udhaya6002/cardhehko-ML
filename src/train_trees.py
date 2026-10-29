"""
train_trees.py - Decision Tree, Random Forest, Gradient Boosting with tuning.

Usage:
    python -m src.train_trees

Rules (SPEC.md):
    - 5-fold KFold, shuffle=True, random_state=42, training set only.
    - No scaling needed for tree models.
    - Report R², RMSE, MAE on both log and rupee scale.
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def _metrics_log(y_true, y_pred):
    return {
        "r2_log": float(r2_score(y_true, y_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
    }


def _metrics_rupee(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "r2_rupee": float(r2_score(y_true, y_pred)),
        "rmse_rupee": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_rupee": float(mean_absolute_error(y_true, y_pred)),
    }


def _evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    log_m = _metrics_log(y_test, y_pred)
    rupee_m = _metrics_rupee(y_test, y_pred)
    print(f"\n  {name} Test Results:")
    print(f"    Log   - R²: {log_m['r2_log']:.4f}, RMSE: {log_m['rmse_log']:.4f}, MAE: {log_m['mae_log']:.4f}")
    print(f"    Rupee - R²: {rupee_m['r2_rupee']:.4f}, RMSE: Rs.{rupee_m['rmse_rupee']:,.0f}, MAE: Rs.{rupee_m['mae_rupee']:,.0f}")
    return {**log_m, **rupee_m}


def train_tree_models(X_train, X_test, y_train, y_test):
    """
    Train Decision Tree, Random Forest, and Gradient Boosting.
    Returns a list of result dicts.
    """
    results = []

    # --- 1. Decision Tree ---
    print("\n[1/3] Decision Tree (GridSearchCV)...")
    dt_params = {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_leaf": [1, 5, 10, 20],
    }
    dt_gs = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        dt_params, cv=CV, scoring="r2", refit=True, n_jobs=-1
    )
    dt_gs.fit(X_train, y_train)
    print(f"  Best params: {dt_gs.best_params_}, CV R²: {dt_gs.best_score_:.4f}")
    test_m = _evaluate(dt_gs.best_estimator_, X_test, y_test, "Decision Tree")
    results.append({
        "model": "Decision Tree",
        "best_params": str(dt_gs.best_params_),
        "cv_r2": float(dt_gs.best_score_),
        **test_m,
    })
    joblib.dump(dt_gs.best_estimator_, "models/decision_tree.joblib")
    with open("results/tree.json", "w") as f:
        json.dump(results[-1], f, indent=2)

    # --- 2. Random Forest ---
    print("\n[2/3] Random Forest (RandomizedSearchCV)...")
    rf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    }
    rf_rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        rf_params, n_iter=20, cv=CV, scoring="r2", refit=True,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_rs.fit(X_train, y_train)
    print(f"  Best params: {rf_rs.best_params_}, CV R²: {rf_rs.best_score_:.4f}")
    test_m = _evaluate(rf_rs.best_estimator_, X_test, y_test, "Random Forest")
    results.append({
        "model": "Random Forest",
        "best_params": str(rf_rs.best_params_),
        "cv_r2": float(rf_rs.best_score_),
        **test_m,
    })
    joblib.dump(rf_rs.best_estimator_, "models/random_forest.joblib")
    with open("results/rf.json", "w") as f:
        json.dump(results[-1], f, indent=2)

    # --- 3. Gradient Boosting ---
    print("\n[3/3] Gradient Boosting (RandomizedSearchCV)...")
    gb_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf": [1, 5, 10],
        "subsample": [0.8, 0.9, 1.0],
    }
    gb_rs = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        gb_params, n_iter=20, cv=CV, scoring="r2", refit=True,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    gb_rs.fit(X_train, y_train)
    print(f"  Best params: {gb_rs.best_params_}, CV R²: {gb_rs.best_score_:.4f}")
    test_m = _evaluate(gb_rs.best_estimator_, X_test, y_test, "Gradient Boosting")
    results.append({
        "model": "Gradient Boosting",
        "best_params": str(gb_rs.best_params_),
        "cv_r2": float(gb_rs.best_score_),
        **test_m,
    })
    joblib.dump(gb_rs.best_estimator_, "models/gradient_boosting.joblib")
    with open("results/gb.json", "w") as f:
        json.dump(results[-1], f, indent=2)

    return results
