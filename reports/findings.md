# Key Findings

1. **Gradient Boosting** achieves the best test R2 of 0.9132 (log scale), explaining ~89.9% of price variance in rupees.
2. Ensemble methods (RF, GB) significantly outperform linear models, suggesting non-linear price-feature relationships.
3. The log-transform of selling_price effectively reduces right-skew, improving model performance across all algorithms.
4. Engineered features (car_age, km_per_year) contribute 0.0428 to R2, confirming their importance.
5. Higher-priced vehicles show larger absolute errors but similar relative error (MAPE), indicating the model scales well across price ranges.