# AQI 24-Hour Prediction - Model Evaluation Report

Comparison between the Random Forest model and a simple persistence baseline (assumes AQI in 24h = AQI right now).

| Metric | Random Forest | Persistence Baseline |
|--------|----------------|------------------------|
| MAE  | 13.56 | 11.24 |
| RMSE | 17.69 | 15.78 |
| R²   | 0.322 | 0.461 |

The Random Forest model did not beat the persistence baseline on MAE. This may mean more training data or additional features (e.g. weather data) are needed.