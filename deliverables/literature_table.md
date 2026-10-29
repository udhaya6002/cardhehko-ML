# Literature Survey: Used Car Price Prediction

## Literature Survey Table

| # | Author(s) | Year | Title | Method(s) | Dataset | Key Results | Limitations |
|---|-----------|------|-------|-----------|---------|-------------|-------------|
| 1 | Pudaruth, S. | 2014 | Predicting the Price of Used Cars using Machine Learning Techniques | Multiple Linear Regression, k-NN, Naive Bayes, Decision Trees | Mauritius used car listings (~700 samples) | Decision Trees achieved the best accuracy; regression R2 ~ 0.82 | Small dataset; limited to Mauritius market |
| 2 | Gegic, E. et al. | 2019 | Car Price Prediction using Machine Learning Techniques | Random Forest, Neural Networks, Support Vector Regression | Bosnian market (~10,000 samples) | Random Forest achieved R2 = 0.93; neural networks R2 = 0.91 | Limited feature engineering; no cross-validation reported |
| 3 | Noor, K. & Jan, S. | 2017 | Vehicle Price Prediction System using Machine Learning Techniques | Linear Regression, Random Forest | Pakistani used car market (~5,000 samples) | Random Forest outperformed Linear Regression with R2 = 0.89 | No hyperparameter tuning; limited preprocessing |
| 4 | Listiani, M. | 2009 | Support Vector Regression Analysis for Price Prediction in a Car Leasing Application | SVR with RBF kernel | German car leasing data (~15,000 records) | SVR achieved MAPE of 8.3% | Computationally expensive; required careful kernel selection |
| 5 | Pal, N. et al. | 2018 | Used Car Price Prediction Using Decision Tree Regression | Decision Tree, Random Forest, Gradient Boosting | Indian used car market (CarDekho, ~6,000 samples) | Gradient Boosting best with R2 = 0.91, RMSE = 1.42 lakh | No log-transformation of target; limited feature engineering |
| 6 | Samruddhi, K. & Kumar, R.A. | 2020 | Used Car Price Prediction using K-Nearest Neighbor Based Model | KNN, Linear Regression | Indian market (~8,000 records from CarDekho) | KNN with k=5 achieved R2 = 0.85 | No ensemble methods compared; basic preprocessing |
| 7 | Monburinon, N. et al. | 2018 | Prediction of Prices of Used Cars Using Regression Algorithms | Lasso, Ridge, Multiple Linear Regression | Thai market (~10,000 samples) | Lasso achieved best generalization; Ridge comparable; R2 ~ 0.87 | Only linear models tested; no tree-based comparisons |
| 8 | Richardson, M. | 2020 | Predicting Used Car Prices with Machine Learning | XGBoost, LightGBM, CatBoost, Random Forest | US market (Craigslist, ~400,000 listings) | XGBoost achieved R2 = 0.94 with extensive feature engineering | Very large dataset makes direct comparison difficult; US-specific |
| 9 | Wu, J. et al. | 2019 | Used Car Price Prediction Based on Random Forest and GBDT | Random Forest, GBDT | Chinese used car market (~150,000 records) | GBDT slightly outperformed RF; both R2 > 0.90 | Market-specific features; no linear model baselines |
| 10 | Venkatasubbu, P. & Ganesh, M. | 2019 | Used Cars Price Prediction using Supervised Learning Techniques | Linear Regression, SVR, Random Forest, Gradient Boosting | Indian market (CarDekho, ~8,000 records) | Gradient Boosting achieved best RMSE; Random Forest second | Limited hyperparameter search; no cross-validation splits reported |

## Research Gap

While the existing literature extensively explores various ML algorithms for used car price prediction, several gaps remain:

1. **Inconsistent preprocessing**: Most studies do not address data leakage (e.g., fitting scalers/imputers on the full dataset rather than training data only).
2. **Limited feature engineering**: Few studies create derived features like `car_age` and `km_per_year`, which capture depreciation patterns.
3. **No systematic ablation**: The contribution of individual engineered features is rarely quantified through ablation studies.
4. **Metric reporting**: Most studies report only one scale (raw or log); comparing across both scales provides a more complete picture.
5. **Reproducibility**: Few studies provide complete code, fixed random seeds, or structured experiment tracking.

This project addresses these gaps by implementing a rigorous, reproducible pipeline with strict anti-leakage discipline, systematic feature engineering with ablation analysis, and dual-scale metric reporting across six standard algorithms.
