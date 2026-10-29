import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Colab Setup & Header
cells.append(nbf.v4.new_markdown_cell("""# Used Car Price Prediction - End-to-End ML Pipeline
**Course Project**: Machine Learning (VIT-AP)  
**Dataset**: CarDekho Used Car Dataset  
**Task**: Regression (Predicting `selling_price` with log1p transformation)  
**Strict Anti-Leakage Compliance**: All scalers, imputers, and percentile caps fit exclusively on training split.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Environment & Dependencies Setup (Colab Compatible)
import sys, os

# If running in Google Colab, clone or mount repo if necessary
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running in Google Colab: {IN_COLAB}")

# Install requirements if in Colab
if IN_COLAB:
    !pip install -q scikit-learn pandas numpy matplotlib seaborn joblib openpyxl kagglehub

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath("..") if os.path.exists("../src") else os.path.abspath("."))
print("System path configured.")
"""))

# Cell 2: Imports & Configuration
cells.append(nbf.v4.new_markdown_cell("""## 2. Imports & Reproducibility Settings"""))
cells.append(nbf.v4.new_code_cell("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set plot styles
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("deep")
RANDOM_STATE = 42
"""))

# Cell 3: Data Loading & Audit
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Ingestion & Audit
Loading `Car details v3.csv` from `data/raw/` and auditing shape, missing values, and data types.
"""))
cells.append(nbf.v4.new_code_cell("""csv_path = "data/raw/Car details v3.csv" if os.path.exists("data/raw/Car details v3.csv") else "../data/raw/Car details v3.csv"
df_raw = pd.read_csv(csv_path)
print(f"Raw Dataset Shape: {df_raw.shape}")
df_raw.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Missing value summary and duplicate counts
print(f"Total duplicates: {df_raw.duplicated().sum()}")
df_raw.info()
"""))

# Cell 4: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering (`src.features`)
Extract numerical values from formatted strings (`mileage`, `engine`, `max_power`, `torque`), derive `brand`, compute vehicle depreciation feature `car_age`, and utilization metric `km_per_year`.
"""))
cells.append(nbf.v4.new_code_cell("""from src.features import engineer_features

# Remove duplicates first
df_clean = df_raw.drop_duplicates()
print(f"After dropping duplicates: {df_clean.shape}")

# Run feature engineering
df_featured = engineer_features(df_clean)
print(f"After feature engineering: {df_featured.shape}")
df_featured.head()
"""))

# Cell 5: Preprocessing Pipeline
cells.append(nbf.v4.new_markdown_cell("""## 5. Strict Anti-Leakage Preprocessing (`src.preprocess`)
1. 80/20 Train/Test split BEFORE any transformations.
2. 1st-99th percentile capping computed ONLY on training set.
3. Median (numeric) & Mode (categorical) imputation fit ONLY on training set.
4. One-hot encoding with dummy category alignment.
"""))
cells.append(nbf.v4.new_code_cell("""from src.preprocess import build_preprocessed_data

X_train, X_test, y_train, y_test = build_preprocessed_data(df_featured, save=False)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
print(f"Zero NaNs in Train: {X_train.isna().sum().sum() == 0}, Zero NaNs in Test: {X_test.isna().sum().sum() == 0}")
"""))

# Cell 6: Model Training
cells.append(nbf.v4.new_markdown_cell("""## 6. Model Training & Hyperparameter Tuning
Evaluating 6 algorithms under 5-Fold Cross Validation:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 regularization tuning)
3. Lasso Regression (L1 regularization & sparsity tuning)
4. Decision Tree Regressor (Depth and leaf tuning)
5. Random Forest Regressor (Ensemble tuning)
6. Gradient Boosting Regressor (Boosting ensemble tuning)
"""))
cells.append(nbf.v4.new_code_cell("""from src.train_linear import train_linear_models
from src.train_trees import train_tree_models

# Train linear models
linear_res = train_linear_models(X_train, X_test, y_train, y_test)

# Train tree models
tree_res = train_tree_models(X_train, X_test, y_train, y_test)

# Model comparison
all_res = linear_res + tree_res
res_df = pd.DataFrame(all_res).sort_values("r2_log", ascending=False)
res_df[["model", "cv_r2", "r2_log", "rmse_log", "mae_log", "r2_rupee", "rmse_rupee", "mae_rupee"]]
"""))

# Cell 7: Visual Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Evaluation & Diagnostics
Plotting comparative model performance, residual distributions, and actual vs predicted values.
"""))
cells.append(nbf.v4.new_code_cell("""# Compare Models
plt.figure(figsize=(10, 5), dpi=150)
sns.barplot(data=res_df, x="r2_log", y="model", palette="viridis")
plt.title("Model Comparison - Test R2 (Log Scale)", fontsize=14, fontweight="bold")
plt.xlabel("R2 Score")
plt.xlim(0.8, 0.95)
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Best Model Residuals & Predictions
best_gb = joblib.load("models/gradient_boosting.joblib" if os.path.exists("models/gradient_boosting.joblib") else "../models/gradient_boosting.joblib")
y_pred = best_gb.predict(X_test)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
axes[0].scatter(y_pred, residuals, alpha=0.3, color="#e74c3c")
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_title("Gradient Boosting - Residual Plot", fontweight="bold")
axes[0].set_xlabel("Predicted log(price)")
axes[0].set_ylabel("Residuals")

axes[1].scatter(y_test, y_pred, alpha=0.3, color="#3498db")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[1].plot(lims, lims, "k--")
axes[1].set_title("Gradient Boosting - Predicted vs Actual", fontweight="bold")
axes[1].set_xlabel("Actual log(price)")
axes[1].set_ylabel("Predicted log(price)")
plt.tight_layout()
plt.show()
"""))

# Cell 8: Feature Importance & Interpretability
cells.append(nbf.v4.new_markdown_cell("""## 8. Interpretability & Feature Importance
Examining top feature drivers identified by the champion Gradient Boosting model.
"""))
cells.append(nbf.v4.new_code_cell("""importances = pd.Series(best_gb.feature_importances_, index=X_train.columns).nlargest(15)

plt.figure(figsize=(10, 6), dpi=150)
importances.sort_values().plot(kind="barh", color="#2ecc71")
plt.title("Top 15 Feature Importances (Gradient Boosting)", fontsize=13, fontweight="bold")
plt.xlabel("Relative Importance")
plt.show()
"""))

# Cell 9: Error Analysis & Ablation Study
cells.append(nbf.v4.new_markdown_cell("""## 9. Error Analysis & Ablation Study
Analyzing error rates across price segments and evaluating the impact of removing engineered features (`car_age`, `km_per_year`).
"""))
cells.append(nbf.v4.new_code_cell("""# Load and display saved error analysis
error_df = pd.read_csv("results/error_analysis.csv" if os.path.exists("results/error_analysis.csv") else "../results/error_analysis.csv")
print("Error Analysis by Segment:")
error_df.head(10)
"""))

cells.append(nbf.v4.new_code_cell("""# Ablation Results
ablation_df = pd.read_csv("results/ablation.csv" if os.path.exists("results/ablation.csv") else "../results/ablation.csv")
print("Feature Ablation Impact:")
ablation_df
"""))

nb.cells = cells

with open("notebooks/main.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Created notebooks/main.ipynb successfully!")
