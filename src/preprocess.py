"""
preprocess.py
--------------
Part of: AirSenseAI (AI-based Real-Time Pollution Prediction and Alert System)

PURPOSE
    Cleans the raw historical Bengaluru AQI data and turns it into a
    model-ready dataset for train_model.py.

    IMPORTANT: The raw dataset already contains a pre-calculated AQI column
    (from CPCB). This script does NOT calculate AQI from pollutant
    concentrations.

    CPCB monitoring stations frequently have missing hourly readings
    (maintenance, sensor downtime, data transmission gaps). To make sure
    lag features and the prediction target reflect real elapsed time --
    not just "the previous row in the file" -- this script rebuilds the
    data onto a continuous hourly timeline before creating them.

INPUT
    data/raw/city_hour.csv

OUTPUT
    data/processed/bengaluru_aqi_model_ready.csv
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/city_hour.csv"
PROCESSED_DATA_PATH = "data/processed/bengaluru_aqi_model_ready.csv"
TARGET_CITY = "Bengaluru"

COLUMNS_TO_KEEP = ["City", "Datetime", "AQI", "AQI_Bucket"]

LAG_HOURS = [1, 3, 6, 12, 24]
TARGET_HOURS_AHEAD = 24


# ---------------------------------------------------------------------------
# 2. LOAD RAW DATA
# ---------------------------------------------------------------------------
def load_raw_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find raw data file at '{filepath}'.")

    print(f"Loading raw data from '{filepath}' ...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} total rows (all cities combined).")
    return df


# ---------------------------------------------------------------------------
# 3. FILTER TO BENGALURU ONLY
# ---------------------------------------------------------------------------
def filter_city(df, city_name):
    df["City"] = df["City"].str.strip()
    city_df = df[df["City"] == city_name].copy()

    print(f"Filtered down to {len(city_df)} rows for '{city_name}'.")
    return city_df


# ---------------------------------------------------------------------------
# 4. SELECT RELEVANT COLUMNS
# ---------------------------------------------------------------------------
def select_columns(df, columns):
    return df[columns]


# ---------------------------------------------------------------------------
# 5. PARSE AND SORT BY DATETIME
# ---------------------------------------------------------------------------
def parse_and_sort_datetime(df):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    print("Converted Datetime to a real date/time type and sorted rows in order.")
    return df


# ---------------------------------------------------------------------------
# 6. REMOVE DUPLICATE ROWS
# ---------------------------------------------------------------------------
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["Datetime"], keep="first")
    after = len(df)

    print(f"Removed {before - after} duplicate rows ({after} rows remain).")
    return df


# ---------------------------------------------------------------------------
# 7. REMOVE INVALID AQI VALUES
# ---------------------------------------------------------------------------
def remove_invalid_aqi(df):
    """
    Keeps only rows where AQI is present and falls inside the official
    CPCB AQI scale (0 to 500). Missing, zero/negative, and unrealistically
    high values are dropped. Any hour removed here is treated the same as
    an hour that was never recorded -- it gets added back as an empty row
    in the next step, instead of keeping a bad value.
    """
    before = len(df)
    df = df[df["AQI"].notna()]
    df = df[(df["AQI"] > 0) & (df["AQI"] <= 500)]
    after = len(df)

    print(f"Removed {before - after} rows with missing/invalid AQI ({after} rows remain).")
    return df


# ---------------------------------------------------------------------------
# 8. REINDEX ONTO A CONTINUOUS HOURLY TIMELINE
# ---------------------------------------------------------------------------
def reindex_to_continuous_hours(df, city_name):
    """
    Rebuilds the dataset so there is exactly one row for every hour
    between the first and last timestamp -- including hours where the
    original data was missing or was removed for being invalid. Those
    hours get empty (NaN) values instead of just disappearing.

    Why this matters: the next steps create lag features and the
    prediction target using .shift(), which moves data by ROW POSITION.
    If an hour were simply missing from the file, every shift after that
    point would silently line up with the wrong number of real hours.
    Making every hour present first means row position and elapsed time
    are guaranteed to match, so the shifts that follow are safe.
    """
    df = df.set_index("Datetime")

    full_hourly_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="H"
    )
    missing_hours = len(full_hourly_range) - len(df)

    df = df.reindex(full_hourly_range)
    df.index.name = "Datetime"
    df["City"] = city_name  # constant for every hour, including filled-in gaps
    df = df.reset_index()

    print(
        f"Expected {len(full_hourly_range)} hourly timestamps from "
        f"{full_hourly_range.min()} to {full_hourly_range.max()}."
    )
    print(
        f"Found {missing_hours} missing hourly timestamps "
        f"({missing_hours / len(full_hourly_range):.1%} of the full range). "
        "These are now present as empty rows so later steps stay accurate."
    )

    return df


# ---------------------------------------------------------------------------
# 9. ADD TIME-BASED FEATURES
# ---------------------------------------------------------------------------
def add_time_features(df):
    df["hour"] = df["Datetime"].dt.hour
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month

    print("Added 'hour', 'day_of_week', and 'month' columns.")
    return df


# ---------------------------------------------------------------------------
# 10. ADD AQI LAG FEATURES
# ---------------------------------------------------------------------------
def add_lag_features(df, lag_hours):
    """
    Adds past-AQI columns (e.g. AQI 1 hour ago, 3 hours ago, ...).
    Safe to do with a simple .shift() now that the timeline has no gaps --
    each shift step corresponds to exactly one real hour.
    """
    for lag in lag_hours:
        df[f"AQI_lag_{lag}h"] = df["AQI"].shift(lag)

    print(f"Added AQI lag features for: {lag_hours} hours.")
    return df


# ---------------------------------------------------------------------------
# 11. ADD 24-HOUR PREDICTION TARGET
# ---------------------------------------------------------------------------
def add_target_column(df, hours_ahead):
    """
    Adds the column the model will actually try to predict: the AQI
    value 'hours_ahead' hours in the future, relative to each row.
    """
    df[f"AQI_target_{hours_ahead}h"] = df["AQI"].shift(-hours_ahead)

    print(f"Added prediction target column: AQI_target_{hours_ahead}h.")
    return df


# ---------------------------------------------------------------------------
# 12. DROP INCOMPLETE ROWS (LAGS / TARGET ONLY)
# ---------------------------------------------------------------------------
def drop_incomplete_rows(df, lag_hours, hours_ahead):
    """
    Drops rows that don't have a full set of lag features or a target
    value. This happens at the start and end of the dataset, where there
    isn't enough history or future data to look at -- and also correctly
    catches any row whose lag/target value would have come from a gap
    hour, since that gap hour is NaN rather than silently pointing at
    the wrong real hour.
    """
    required_columns = [f"AQI_lag_{lag}h" for lag in lag_hours]
    required_columns.append(f"AQI_target_{hours_ahead}h")

    before = len(df)
    df = df.dropna(subset=required_columns)
    after = len(df)

    print(f"Dropped {before - after} incomplete rows ({after} rows remain).")
    return df


# ---------------------------------------------------------------------------
# 13. SAVE PROCESSED DATA
# ---------------------------------------------------------------------------
def save_processed_data(df, filepath):
    output_folder = os.path.dirname(filepath)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"Saved model-ready data to '{filepath}' ({len(df)} rows).")


# ---------------------------------------------------------------------------
# 14. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    df = load_raw_data(RAW_DATA_PATH)
    df = filter_city(df, TARGET_CITY)
    df = select_columns(df, COLUMNS_TO_KEEP)
    df = parse_and_sort_datetime(df)
    df = remove_duplicates(df)
    df = remove_invalid_aqi(df)
    df = reindex_to_continuous_hours(df, TARGET_CITY)
    df = add_time_features(df)
    df = add_lag_features(df, LAG_HOURS)
    df = add_target_column(df, TARGET_HOURS_AHEAD)
    df = drop_incomplete_rows(df, LAG_HOURS, TARGET_HOURS_AHEAD)
    save_processed_data(df, PROCESSED_DATA_PATH)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()