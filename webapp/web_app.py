
# Use official Python slim image
FROM python:3.9-slim
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libatlas-base-dev libffi-dev libssl-dev libpq-dev python3-dev curl \
    && rm -rf /var/lib/apt/lists/*
# Install mlflow
RUN pip install --no-cache-dir mlflow[extras]
# Create app directory
WORKDIR /app
# Expose MLflow port
EXPOSE 5003
# Set environment variables if needed (optional)
ENV MLFLOW_HOME=/app
# Start MLflow server with SQLite backend and local artifact store
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlruns \
    --host 0.0.0.0 \
    --port 5003
(venv) root@lutov1:~/WePP_Project_copy# cd webapp
(venv) root@lutov1:~/WePP_Project_copy/webapp# cat web_app.py
from flask import Flask, render_template, request
import mlflow
import mlflow.pyfunc
import pandas as pd
import os

app = Flask(__name__)

# Set the MLflow tracking URI to your MLflow server container address
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5003")
print(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Model details
model_name = "FlightPrice"
model_uri = f"models:/{model_name}/Production"
print(f"Loading model from: {model_uri}")

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None  # Prevent crash, handle gracefully later

# Extract feature names from model metadata if model loaded
model_feature_names = []
if model is not None:
    model_input_schema = model.metadata.get_input_schema()
    model_feature_names = [field.name for field in model_input_schema.inputs]

# Cities shown to the user
CITIES = ["Zurich", "Prague", "Copenhagen", "Barcelona", "Rome", "Paris"]

# Mapping display names to DB codes
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
    origin = destination = flight_date_str = None

    if request.method == "POST":
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        flight_date_str = request.form.get("flight_date")

        if origin == destination:
            error = "Departure and destination cities cannot be the same."
        elif model is None:
            error = "Model is not loaded; prediction not possible."
        else:
            try:
                # Convert city names to codes
                origin_code = CITY_TO_CODE.get(origin)
                destination_code = CITY_TO_CODE.get(destination)

                # Process date
                flight_date = pd.to_datetime(flight_date_str)
                reference_date = pd.to_datetime("2025-05-05")
                days_until = (flight_date - reference_date).days
                weekday = float(flight_date.dayofweek)
                is_weekend = int(weekday >= 5)

                # Prepare input dataframe
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

                # One-hot encode input
                input_encoded = pd.get_dummies(input_data)

                # Align columns with model features
                for col in model_feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                input_encoded = input_encoded[model_feature_names]

                # Correct dtypes for numeric columns
                numeric_cols_float = ['days_until_flight', 'departure_hour', 'departure_weekday', 'is_weekend']
                for col in numeric_cols_float:
                    if col in input_encoded.columns:
                        input_encoded[col] = input_encoded[col].fillna(0.0).astype('float64')

                # Convert categorical columns to bool
                categorical_cols = list(set(model_feature_names) - set(numeric_cols_float))
                for col in categorical_cols:
                    if col in input_encoded.columns:
                        input_encoded[col] = input_encoded[col].astype(bool)

                # DEBUG: print input info before prediction
                print(input_encoded.dtypes)
                print(input_encoded.isnull().sum())
                print(input_encoded.head())

                # Predict
                prediction = model.predict(input_encoded)[0]
                print(f"Predicted price: CHF {prediction:.2f}")

            except Exception as e:
                error = f"Error during prediction: {e}"
                print(error)

    return render_template("form.html",
                           cities=CITIES,
                           prediction=prediction,
                           error=error,
                           origin=origin,
                           destination=destination,
                           flight_date=flight_date_str)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)