# Decision Log

- **Dataset**: Car details v3.csv chosen (8128 rows x 13 cols, richest feature set).
- **Duplicates**: Dropped (exact row duplicates removed before processing).
- **Torque parsing**: Converted kgm to Nm (x9.80665); extracted max RPM from ranges.
- **Brand extraction**: First word of car name, title-cased.
- **Rare brands**: Kept all brands; one-hot encoding handles low-frequency brands.
- **car_age**: 2024 - year (fixed reference year for reproducibility).
- **km_per_year**: km_driven / car_age (NaN if age=0).
- **Outlier capping**: 1st-99th percentile on training set only.
- **Imputation**: Median for numeric, mode for categorical (fit on training set).
- **Encoding**: One-hot with drop_first=True; test set aligned to training columns.
- **Scaling**: StandardScaler for linear models only (fit on training set).
- **Target transform**: log1p(selling_price) to reduce skew.

## Key Results

- **Best model**: Gradient Boosting with test R2=0.9132 (log), R2=0.8987 (rupee)
- **Best RMSE (log)**: 0.2201
- **Best RMSE (rupee)**: Rs.149,075
- **Ablation**: Removing car_age and km_per_year drops R2 from 0.9132 to 0.8704 -> engineered features add value.