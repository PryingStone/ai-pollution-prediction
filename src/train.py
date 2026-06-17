"""
train.py
--------
Part of: AirSenseAI (AI-based Real-Time Pollution Prediction and Alert System)

PURPOSE
    Trains a RandomForestRegressor to predict Bengaluru's AQI 24 hours
    ahead, using the model-ready dataset produced by preprocess.py.

    The split between training and testing data is done by TIME, not at
    random -- the most recent slice of the timeline is held out as the
    test set. This matches how the model will actually be used (predicting
    the future from the past) and avoids letting the model "peek" at
    future data during training.

    The Random Forest is compared against a persistence baseline
    (assume AQI in 24h = AQI right now), so we have a simple, honest way
    to check whether the model is actually learning something useful.

INPUT
    data/processed/bengaluru_aqi_model_ready.csv

OUTPUT
    models/random_forest_aqi24h.joblib   (the trained model)
    models/feature_list.json             (the exact feature columns used)
    models/metrics_report.md             (Random Forest vs baseline scores)
    models/feature_importance.csv        (how much each feature contributed)
"""

import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH = "data/processed/bengaluru_aqi_model_ready.csv"

MODEL_PATH = "models/random_forest_aqi24h.joblib"
FEATURE_LIST_PATH = "models/feature_list.json"
METRICS_REPORT_PATH = "models/metrics_report.md"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.csv"

# These must match the columns preprocess.py actually creates.
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "AQI",
    "AQI_lag_1h",
    "AQI_lag_3h",
    "AQI_lag_6h",
    "AQI_lag_12h",
    "AQI_lag_24h",
]
TARGET_COLUMN = "AQI_target_24h"

TEST_SIZE_RATIO = 0.2  # the most recent 20% of the timeline is held out for testing

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,      # more trees = more stable predictions
    "max_depth": 15,          # limits how deep each tree can grow, reduces overfitting
    "min_samples_split": 5,   # a node must have at least 5 samples before it can be split
    "n_jobs": -1,             # use all available CPU cores to speed up training
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# 2. LOAD PROCESSED DATA
# ---------------------------------------------------------------------------
def load_processed_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find processed data file at '{filepath}'. "
            "Run preprocess.py first."
        )

    print(f"Loading processed data from '{filepath}' ...")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    print(f"Loaded {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# 3. TIME-BASED TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
def time_based_train_test_split(df, test_size_ratio):
    """
    Splits the data by time instead of randomly. The earlier portion of
    the timeline is used for training, and the most recent portion is
    held out for testing -- the same way the model will be used in
    practice (always predicting forward, never backward).
    """
    split_index = int(len(df) * (1 - test_size_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print(
        f"Train set: {len(train_df)} rows "
        f"({train_df['Datetime'].min()} to {train_df['Datetime'].max()})."
    )
    print(
        f"Test set: {len(test_df)} rows "
        f"({test_df['Datetime'].min()} to {test_df['Datetime'].max()})."
    )

    return train_df, test_df


# ---------------------------------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# ---------------------------------------------------------------------------
def prepare_features_and_target(df, feature_columns, target_column):
    """
    Splits a dataframe into the X (features) the model learns from and
    the y (target) it tries to predict. Also drops any leftover rows
    with missing values as a safety net, even though preprocess.py
    should have already handled this.
    """
    df = df.dropna(subset=feature_columns + [target_column])
    X = df[feature_columns]
    y = df[target_column]
    return X, y


# ---------------------------------------------------------------------------
# 5. TRAIN RANDOM FOREST
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train, params):
    print("Training RandomForestRegressor ...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


# ---------------------------------------------------------------------------
# 6. PERSISTENCE BASELINE
# ---------------------------------------------------------------------------
def get_baseline_predictions(df):
    """
    The simplest possible forecast: assume AQI 24 hours from now will be
    the same as AQI right now. Any model we build should be able to beat
    this baseline -- if it can't, the model isn't adding real value.
    """
    return df["AQI"]


# ---------------------------------------------------------------------------
# 7. EVALUATE PREDICTIONS
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"{label} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# 8. SAVE MODEL
# ---------------------------------------------------------------------------
def save_model(model, filepath):
    output_folder = os.path.dirname(filepath)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"Saved trained model to '{filepath}'.")


# ---------------------------------------------------------------------------
# 9. SAVE FEATURE LIST
# ---------------------------------------------------------------------------
def save_feature_list(feature_columns, filepath):
    """
    Saves the exact list and order of feature columns the model was
    trained on. predict.py will need this later to build inputs in the
    same shape the model expects.
    """
    output_folder = os.path.dirname(filepath)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"Saved feature list to '{filepath}'.")


# ---------------------------------------------------------------------------
# 10. SAVE METRICS REPORT
# ---------------------------------------------------------------------------
def save_metrics_report(rf_metrics, baseline_metrics, filepath):
    output_folder = os.path.dirname(filepath)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    lines = [
        "# AQI 24-Hour Prediction - Model Evaluation Report",
        "",
        "Comparison between the Random Forest model and a simple "
        "persistence baseline (assumes AQI in 24h = AQI right now).",
        "",
        "| Metric | Random Forest | Persistence Baseline |",
        "|--------|----------------|------------------------|",
        f"| MAE  | {rf_metrics['MAE']:.2f} | {baseline_metrics['MAE']:.2f} |",
        f"| RMSE | {rf_metrics['RMSE']:.2f} | {baseline_metrics['RMSE']:.2f} |",
        f"| R\u00b2   | {rf_metrics['R2']:.3f} | {baseline_metrics['R2']:.3f} |",
        "",
    ]

    if rf_metrics["MAE"] < baseline_metrics["MAE"]:
        lines.append(
            "The Random Forest model beats the persistence baseline on "
            "MAE, meaning it is learning something useful beyond simply "
            "assuming AQI stays the same for 24 hours."
        )
    else:
        lines.append(
            "The Random Forest model did not beat the persistence "
            "baseline on MAE. This may mean more training data or "
            "additional features (e.g. weather data) are needed."
        )

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved metrics report to '{filepath}'.")


# ---------------------------------------------------------------------------
# 11. SAVE FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
def save_feature_importance(model, feature_columns, filepath):
    """
    After training, a Random Forest can tell us how much each feature
    contributed to its predictions. Higher importance means the model
    relied on that feature more when making decisions.

    This is saved as a CSV so you can inspect it or plot it later --
    useful for understanding what the model is actually paying attention to
    and for deciding which features to add or drop in future iterations.
    """
    output_folder = os.path.dirname(filepath)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(filepath, index=False)
    print(f"Saved feature importances to '{filepath}'.")
    print(importance_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 12. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    df = load_processed_data(PROCESSED_DATA_PATH)
    train_df, test_df = time_based_train_test_split(df, TEST_SIZE_RATIO)

    X_train, y_train = prepare_features_and_target(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = prepare_features_and_target(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    model = train_random_forest(X_train, y_train, RANDOM_FOREST_PARAMS)
    rf_predictions = model.predict(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_predictions, "Random Forest")

    # Use the same rows as the test set (after the safety-net dropna above)
    # so the baseline is judged on an identical set of rows as the model.
    baseline_predictions = get_baseline_predictions(test_df.loc[y_test.index])
    baseline_metrics = evaluate_predictions(y_test, baseline_predictions, "Persistence Baseline")

    save_model(model, MODEL_PATH)
    save_feature_list(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    save_metrics_report(rf_metrics, baseline_metrics, METRICS_REPORT_PATH)
    save_feature_importance(model, FEATURE_COLUMNS, FEATURE_IMPORTANCE_PATH)

    print("Training and evaluation complete.")


if __name__ == "__main__":
    main()