"""
predict.py
----------
Part of: AirSenseAI (AI-based Real-Time Pollution Prediction and Alert System)

PURPOSE
    Uses the trained Random Forest model to predict Bengaluru's AQI
    24 hours from now, based on the current live AQI fetched from the
    WAQI (World Air Quality Index) API.

    Steps:
        1. Load the saved model and feature list from the models/ folder.
        2. Fetch the current AQI for Bengaluru from the WAQI API.
        3. Build a feature row that matches exactly what the model was
           trained on (same columns, same order).
        4. Run the prediction.
        5. Return a result dictionary with the current AQI, predicted AQI,
           how much it is expected to change, and a risk category label.

INPUT
    models/random_forest_aqi24h.joblib   (trained model)
    models/feature_list.json             (feature columns the model expects)
    WAQI_API_KEY environment variable    (your WAQI API token)

OUTPUT
    A Python dictionary with prediction results, printed to the terminal.
    This dictionary is also what dashboard.py will import and call.

HOW TO RUN
    Set your API key first, then run the script:

        On Mac/Linux:
            export WAQI_API_KEY="your_token_here"
            python src/predict.py

        On Windows (Command Prompt):
            set WAQI_API_KEY=your_token_here
            python src/predict.py

    Get a free WAQI API token at: https://aqicn.org/data-platform/token/
"""

import os
import json
import joblib
import datetime

from dotenv import load_dotenv

import requests
import pandas as pd

load_dotenv()
# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_PATH        = "models/random_forest_aqi24h.joblib"
FEATURE_LIST_PATH = "models/feature_list.json"

# The WAQI station identifier for Bengaluru.
# "@7021" is the WAQI station ID for Bengaluru. You can also use the
# city keyword "bengaluru" but a station ID is more reliable.
WAQI_STATION = "@7021"

# AQI risk categories based on India's National AQI standard (CPCB).
# Each entry is (upper_limit, category_label, advice).
AQI_CATEGORIES = [
    (50,  "Good",           "Air quality is satisfactory. Enjoy outdoor activities."),
    (100, "Satisfactory",   "Air quality is acceptable. Sensitive individuals should limit prolonged exertion outdoors."),
    (200, "Moderate",       "Sensitive groups may experience discomfort. Consider reducing outdoor activity."),
    (300, "Poor",           "Everyone may begin to experience health effects. Limit outdoor activity."),
    (400, "Very Poor",      "Health warnings. Avoid prolonged outdoor exertion. Use a mask outdoors."),
    (500, "Severe",         "Health alert: everyone may experience serious health effects. Stay indoors."),
]


# ---------------------------------------------------------------------------
# 2. LOAD MODEL AND FEATURE LIST
# ---------------------------------------------------------------------------
def load_model(model_path):
    """
    Loads the trained Random Forest model from disk.
    The model was saved by train.py using joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run train.py first to generate the model."
        )
    print(f"Loading model from '{model_path}' ...")
    model = joblib.load(model_path)
    print("Model loaded.")
    return model


def load_feature_list(feature_list_path):
    """
    Loads the list of feature columns the model was trained on.
    We need this to make sure the prediction input has exactly the same
    columns in exactly the same order -- otherwise the model will produce
    wrong results or raise an error.
    """
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(
            f"Feature list not found at '{feature_list_path}'. "
            "Please run train.py first to generate the feature list."
        )
    with open(feature_list_path, "r") as f:
        feature_list = json.load(f)
    print(f"Feature list loaded: {feature_list}")
    return feature_list


# ---------------------------------------------------------------------------
# 3. FETCH LIVE AQI FROM WAQI
# ---------------------------------------------------------------------------
def fetch_live_aqi(station, api_key):
    """
    Calls the WAQI API to get the current AQI reading for Bengaluru.

    The API returns a JSON response. We pull out:
        - aqi:  the current AQI value (integer)
        - time: the timestamp of the reading

    WAQI API docs: https://aqicn.org/json-api/doc/
    """
    if not api_key:
        raise ValueError(
            "WAQI API key is missing. "
            "Set the environment variable WAQI_API_KEY before running this script.\n"
            "Example (Mac/Linux): export WAQI_API_KEY='your_token_here'\n"
            "Example (Windows):   set WAQI_API_KEY=your_token_here\n"
            "Get a free token at: https://aqicn.org/data-platform/token/"
        )

    url = f"https://api.waqi.info/feed/{station}/?token={api_key}"
    print(f"Fetching live AQI from WAQI for station '{station}' ...")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises an error for HTTP 4xx/5xx responses
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to the WAQI API. "
            "Please check your internet connection and try again."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "The WAQI API request timed out after 10 seconds. "
            "The server may be slow. Please try again."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"WAQI API returned an HTTP error: {e}")

    data = response.json()

    # The WAQI API wraps its response in a status field.
    if data.get("status") != "ok":
        raise RuntimeError(
            f"WAQI API returned an error status: {data.get('data', 'Unknown error')}. "
            "Check that your API key is valid and the station ID is correct."
        )

    aqi_value = data["data"]["aqi"]
    reading_time = data["data"]["time"]["s"]  # e.g. "2024-03-15 14:00:00"

    # WAQI occasionally returns "-" instead of a number when data is unavailable.
    if not isinstance(aqi_value, (int, float)):
        raise ValueError(
            f"WAQI returned a non-numeric AQI value: '{aqi_value}'. "
            "The station may not have a current reading. Try again later."
        )

    print(f"Live AQI fetched: {aqi_value} (reading time: {reading_time})")
    return int(aqi_value), reading_time


# ---------------------------------------------------------------------------
# 4. BUILD FEATURE ROW
# ---------------------------------------------------------------------------
def build_feature_row(current_aqi, feature_list):
    """
    Constructs the single row of input features that the model needs to
    make a prediction.

    The model was trained on these columns (from feature_list.json):
        - hour, day_of_week, month   : time-based features
        - AQI                        : the current AQI reading
        - AQI_lag_1h through _24h    : AQI values from 1, 3, 6, 12, 24 hours ago

    Since this is a live prediction (not a historical one), we do not have
    the real lag values. The best we can do is fill them all with the
    current AQI. This is the same assumption as the persistence baseline --
    it is a reasonable approximation when the last few hours of AQI data
    are not cached locally.

    NOTE: Once weather features are added in a later version, they will
    also be constructed here.
    """
    now = datetime.datetime.now()

    # Build a dictionary with every feature the model expects.
    feature_values = {}
    for feature in feature_list:
        if feature == "hour":
            feature_values["hour"] = now.hour
        elif feature == "day_of_week":
            # Monday = 0, Sunday = 6 (matches pandas/sklearn convention)
            feature_values["day_of_week"] = now.weekday()
        elif feature == "month":
            feature_values["month"] = now.month
        elif feature == "AQI":
            feature_values["AQI"] = current_aqi
        elif feature.startswith("AQI_lag_"):
            # Fill all lag features with current AQI as best approximation.
            feature_values[feature] = current_aqi
        else:
            # Unknown feature -- fill with 0 and warn.
            print(f"Warning: unknown feature '{feature}' filled with 0.")
            feature_values[feature] = 0

    # Wrap in a DataFrame so the model receives the right shape and column names.
    feature_row = pd.DataFrame([feature_values])[feature_list]
    return feature_row


# ---------------------------------------------------------------------------
# 5. GET AQI RISK CATEGORY
# ---------------------------------------------------------------------------
def get_aqi_category(aqi_value):
    """
    Maps a numeric AQI value to a risk category label and health advice,
    based on India's CPCB (Central Pollution Control Board) AQI standard.

    The six categories are:
        0-50    Good
        51-100  Satisfactory
        101-200 Moderate
        201-300 Poor
        301-400 Very Poor
        401+    Severe
    """
    for upper_limit, category, advice in AQI_CATEGORIES:
        if aqi_value <= upper_limit:
            return category, advice

    # If AQI is above 500 (rare but possible during severe pollution events)
    return "Severe", AQI_CATEGORIES[-1][2]


# ---------------------------------------------------------------------------
# 6. RUN PREDICTION
# ---------------------------------------------------------------------------
def run_prediction():
    """
    Main function that ties everything together:
        1. Load model and feature list.
        2. Fetch current AQI.
        3. Build the feature row.
        4. Predict AQI 24 hours ahead.
        5. Compute change and risk categories.
        6. Return a result dictionary.

    Returns a dictionary with:
        current_aqi        : int   - live AQI right now
        predicted_aqi      : int   - model's 24h forecast
        aqi_change         : int   - predicted_aqi minus current_aqi
        current_category   : str   - risk label for current AQI
        current_advice     : str   - health advice for current AQI
        predicted_category : str   - risk label for predicted AQI
        predicted_advice   : str   - health advice for predicted AQI
        reading_time       : str   - timestamp of the live AQI reading
        prediction_time    : str   - local time when this prediction was made
    """
    # --- Load model artifacts ---
    model = load_model(MODEL_PATH)
    feature_list = load_feature_list(FEATURE_LIST_PATH)

    # --- Fetch live AQI ---
    api_key = os.environ.get("WAQI_API_KEY", "")
    current_aqi, reading_time = fetch_live_aqi(WAQI_STATION, api_key)

    # --- Build input and predict ---
    feature_row = build_feature_row(current_aqi, feature_list)
    predicted_aqi = int(round(model.predict(feature_row)[0]))

    # --- Derive supporting information ---
    aqi_change = predicted_aqi - current_aqi
    current_category, current_advice     = get_aqi_category(current_aqi)
    predicted_category, predicted_advice = get_aqi_category(predicted_aqi)
    prediction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "current_aqi":        current_aqi,
        "predicted_aqi":      predicted_aqi,
        "aqi_change":         aqi_change,
        "current_category":   current_category,
        "current_advice":     current_advice,
        "predicted_category": predicted_category,
        "predicted_advice":   predicted_advice,
        "reading_time":       reading_time,
        "prediction_time":    prediction_time,
    }

    return result


# ---------------------------------------------------------------------------
# 7. PRINT RESULT (when run directly from terminal)
# ---------------------------------------------------------------------------
def print_result(result):
    """
    Displays the prediction result in a readable format in the terminal.
    When dashboard.py imports and calls run_prediction(), it will use the
    dictionary directly without printing.
    """
    change_sign = "+" if result["aqi_change"] >= 0 else ""

    print("\n" + "=" * 50)
    print("  AirSenseAI -- 24-Hour AQI Forecast")
    print("=" * 50)
    print(f"  Prediction made at : {result['prediction_time']}")
    print(f"  AQI reading time   : {result['reading_time']}")
    print("-" * 50)
    print(f"  Current AQI        : {result['current_aqi']}  ({result['current_category']})")
    print(f"  Predicted AQI      : {result['predicted_aqi']}  ({result['predicted_category']})")
    print(f"  Expected change    : {change_sign}{result['aqi_change']}")
    print("-" * 50)
    print(f"  Now    : {result['current_advice']}")
    print(f"  In 24h : {result['predicted_advice']}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        result = run_prediction()
        print_result(result)
    except (FileNotFoundError, ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"\nError: {e}\n")