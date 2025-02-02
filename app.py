from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Load everything from "model_data.pkl"
model_data = joblib.load("model_data.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
mae = model_data["mae"]  # ✅ Fix: Now correctly loading MAE
feature_order = model_data["feature_order"]

@app.route("/")
def home():
    """Render the homepage with a form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Receive form input, process it, and return the predicted price with confidence range."""
    try:
        # Extract user input
        input_data = {
            "NO_BEDROOMS": float(request.form["NO_BEDROOMS"]),
            "SQUARE_FEET": float(request.form["SQUARE_FEET"]),
            "HOME_AGE": float(request.form["HOME_AGE"]),
            "GARAGE_SPACES_CC": float(request.form["GARAGE_SPACES_CC"]),
            "FULL_BATHS": float(request.form["FULL_BATHS"]),
            "HALF_BATHS": float(request.form["HALF_BATHS"])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Add missing categorical features (set them to 0)
        for col in feature_order:
            if col not in input_df.columns:
                input_df[col] = 0  # Default missing categorical features to 0

        # Ensure input follows the exact feature order from training
        input_df = input_df[feature_order]

        # Scale numerical features
        num_features = ["NO_BEDROOMS", "SQUARE_FEET", "HOME_AGE", "GARAGE_SPACES_CC", "FULL_BATHS", "HALF_BATHS"]
        input_df[num_features] = scaler.transform(input_df[num_features])

        # Make the prediction
        predicted_price = model.predict(input_df)[0]

        # Calculate confidence range
        lower_bound = predicted_price - mae
        upper_bound = predicted_price + mae

        return render_template(
            "index.html",
            prediction=f"Estimated Condo Price: ${predicted_price:,.2f}",
            confidence=f"Confidence Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}"
        )

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
