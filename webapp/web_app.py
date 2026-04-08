from flask import Flask, render_template, request
import pandas as pd
import os
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "model", "best_model.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "src", "model", "feature_names.pkl")

model = None
model_feature_names = []
model_load_error = None

# Load model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        model_load_error = f"Could not load model file: {e}"
        print(f"❌ {model_load_error}")
else:
    model_load_error = (
        "Model file not found. Add 'best_model.pkl' to src/model/ "
        "if you want to run predictions locally."
    )
    print(f"⚠️ {model_load_error}")

# Load feature names
if os.path.exists(FEATURES_PATH):
    try:
        model_feature_names = joblib.load(FEATURES_PATH)
        print("✅ Feature names loaded successfully.")
    except Exception as e:
        if model_load_error:
            model_load_error += f" Also failed to load feature names: {e}"
        else:
            model_load_error = f"Could not load feature names: {e}"
        print(f"❌ {model_load_error}")
else:
    if model_load_error:
        model_load_error += " Feature names file not found."
    else:
        model_load_error = "Feature names file not found."
    print(f"⚠️ {model_load_error}")


CITIES = ["Zurich", "Prague", "Copenhagen", "Barcelona", "Rome", "Paris"]

CITY_TO_CODE = {
    "Zurich": "ZRH",
    "Prague": "PRG",
    "Copenhagen": "CPH",
    "Barcelona": "BCN",
    "Rome": "FCO",
    "Paris": "CDG"
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    origin = None
    destination = None
    flight_date_str = None

    if request.method == "POST":
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        flight_date_str = request.form.get("flight_date")

        if origin == destination:
            error = "Departure and destination cities cannot be the same."

        elif model is None:
            error = model_load_error or (
                "Model is not available. Add best_model.pkl to src/model/ "
                "to enable predictions."
            )

        elif not model_feature_names:
            error = "Feature names are missing. Add feature_names.pkl to src/model/."

        else:
            try:
                origin_code = CITY_TO_CODE.get(origin)
                destination_code = CITY_TO_CODE.get(destination)

                if not origin_code or not destination_code:
                    raise ValueError("Invalid city selection.")

                flight_date = pd.to_datetime(flight_date_str)
                reference_date = pd.to_datetime("2025-05-05")

                days_until = (flight_date - reference_date).days
                weekday = float(flight_date.dayofweek)
                is_weekend = int(weekday >= 5)

                input_data = pd.DataFrame([{
                    "airline": "Swiss",
                    "number_of_stops": "Direct",
                    "origin": origin_code,
                    "destination": destination_code,
                    "days_until_flight": days_until,
                    "departure_hour": 12.0,
                    "departure_weekday": weekday,
                    "is_weekend": is_weekend
                }])

                input_encoded = pd.get_dummies(input_data)

                for col in model_feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0

                input_encoded = input_encoded[model_feature_names]

                numeric_cols = [
                    "days_until_flight",
                    "departure_hour",
                    "departure_weekday",
                    "is_weekend"
                ]

                for col in numeric_cols:
                    if col in input_encoded.columns:
                        input_encoded[col] = input_encoded[col].fillna(0.0).astype("float64")

                categorical_cols = list(set(model_feature_names) - set(numeric_cols))
                for col in categorical_cols:
                    if col in input_encoded.columns:
                        input_encoded[col] = input_encoded[col].astype(bool)

                prediction = model.predict(input_encoded)[0]
                print(f"Predicted price: CHF {prediction:.2f}")

            except Exception as e:
                error = f"Prediction error: {e}"
                print(error)

    return render_template(
        "form.html",
        cities=CITIES,
        prediction=prediction,
        error=error,
        origin=origin,
        destination=destination,
        flight_date=flight_date_str,
        model_available=(model is not None)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
