# Project Spec

## Setting & Value

| Setting | Value |
|---------|-------|
| **Project** | Used car price prediction (regression) for the ML course project at VIT-AP |
| **Dataset** | CarDekho used car dataset (Kaggle); confirm the exact file name after download and record rows and columns in reports/data_audit.md |
| **Task** | Regression |
| **Target** | `selling_price`, modelled as `log1p(selling_price)`; report metrics on both log and rupee scale |
| **Random seed** | 42 (everywhere) |
| **Train/test split** | 80/20, done BEFORE any imputation, capping or scaling |
| **Cross-validation** | 5-fold KFold, `shuffle=True`, `random_state=42`, on the training set only |
| **Models** | Linear Regression (baseline), Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting |
| **Metrics** | R², RMSE, MAE (log and rupee scale) |
| **Preprocessing** | Parse unit strings to numbers; brand from name; car_age; km_per_year; median/mode imputation; 1st–99th percentile capping on training data; one-hot encoding; scaling for linear models only |
| **Libraries** | Python 3, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, joblib, Jupyter/Colab |
| **Leakage rule** | Every transformer, cap threshold and imputer is fit on training data only |
| **Reproducibility** | Every script runs from the repo root; `notebooks/main.ipynb` runs top to bottom |
| **Figures** | Saved to `figures/` at 200 dpi, titled, axes labelled |
| **Decision log** | Every decision and key number goes in `reports/decisions.md` as one line |
| **Do not** | Edit `data/raw/`, hard-code numbers into the report, or tune on the test set |
