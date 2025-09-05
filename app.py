# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, Tuple

app = Flask(__name__)
MODEL_PATH = "phone_price_predictor.pkl"

_model = None

def load_model():
    """Lazily load the saved joblib pipeline."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Please create and save your pipeline to this path."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

def build_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Construct a single-row DataFrame with the feature names expected by the model."""
    columns = [
        "ram",
        "storage",
        "battery_mAh",
        "charging_watt",
        "display_size",
        "refresh_rate",
        "rear_camera",
        "brand",
    ]

    # Coerce numeric values; brand keep as string (lowercased)
    row = {
        "ram": float(payload.get("ram", 0)),
        "storage": float(payload.get("storage", 0)),
        "battery_mAh": float(payload.get("battery_mAh", 0)),
        "charging_watt": float(payload.get("charging_watt", 0)),
        "display_size": float(payload.get("display_size", 0)),
        "refresh_rate": float(payload.get("refresh_rate", 0)),
        "rear_camera": float(payload.get("rear_camera", 0)),
        "brand": str(payload.get("brand", "")).strip().lower(),
    }
    return pd.DataFrame([row], columns=columns)

# Defaults used for form initial render and for fallback parsing
DEFAULTS = {
    "ram": 8,
    "storage": 128,
    "battery_mAh": 4000,
    "charging_watt": 18,
    "display_size": 6.2,
    "refresh_rate": 60,
    "rear_camera": 48,
    "brand": "samsung",
}

def parse_form(form: Dict[str, Any], defaults: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse request.form into two dicts:
      - inputs_ui: values for populating the form (user-friendly/display)
      - payload_for_model: normalized values passed to build_input_df/model
    Returns (inputs_ui, payload_for_model)
    """
    def parse_int(name, default):
        raw = form.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(float(raw))
        except Exception:
            return default

    def parse_float(name, default):
        raw = form.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except Exception:
            return default

    # Keep the raw display of brand typed by user if they used "Others"
    selected_brand = (form.get("brand") or defaults["brand"]).strip()
    other_brand_raw = (form.get("other_brand") or "").strip()

    # If user selected 'others' and provided an other_brand, use that for display/model
    if selected_brand.lower() == "others":
        brand_display = other_brand_raw if other_brand_raw else "Others"
        model_brand = other_brand_raw.lower() if other_brand_raw else "others"
    else:
        brand_display = selected_brand
        model_brand = selected_brand.lower()

    inputs_ui = {
        "ram": parse_int("ram", defaults["ram"]),
        "storage": parse_int("storage", defaults["storage"]),
        "battery_mAh": parse_int("battery_mAh", defaults["battery_mAh"]),
        "charging_watt": parse_int("charging_watt", defaults["charging_watt"]),
        "display_size": parse_float("display_size", defaults["display_size"]),
        "refresh_rate": parse_int("refresh_rate", defaults["refresh_rate"]),
        "rear_camera": parse_int("rear_camera", defaults["rear_camera"]),
        "brand": brand_display,
        "other_brand": other_brand_raw,
    }

    payload_for_model = {
        "ram": inputs_ui["ram"],
        "storage": inputs_ui["storage"],
        "battery_mAh": inputs_ui["battery_mAh"],
        "charging_watt": inputs_ui["charging_watt"],
        "display_size": inputs_ui["display_size"],
        "refresh_rate": inputs_ui["refresh_rate"],
        "rear_camera": inputs_ui["rear_camera"],
        "brand": model_brand,
    }

    return inputs_ui, payload_for_model

@app.route("/", methods=["GET", "POST"])
def index():
    """Render form and handle prediction requests."""
    error = None
    result = None

    # by default show defaults in the form
    inputs = { **DEFAULTS, "other_brand": "" }

    if request.method == "POST":
        # parse form into UI inputs and model payload (with fallbacks)
        inputs, payload = parse_form(request.form, DEFAULTS)

        try:
            # Build DataFrame for model and make prediction
            df = build_input_df(payload)
            model = load_model()

            # Model expects log1p transformed target at training => model.predict returns log(price)
            y_log_pred = model.predict(df)
            y_pred = np.expm1(y_log_pred)  # invert log1p

            predicted_price = float(y_pred[0])
            result = {"predicted_price": round(predicted_price, 2)}
        except Exception as exc:
            # show error to user (in production you might log this instead)
            error = (
                "Prediction failed: "
                + (str(exc) or "unknown error")
            )

    # Render the template with the inputs dict (either defaults or submitted values),
    # so the form re-populates with the user's last submitted values.
    return render_template("index.html", result=result, error=error, inputs=inputs)


@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Programmatic JSON endpoint."""
    try:
        payload = request.get_json(force=True)
        if not payload or not isinstance(payload, dict):
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        # Required minimal fields
        required = {"ram", "storage", "battery_mAh", "charging_watt", "display_size", "refresh_rate", "rear_camera", "brand"}
        if not required.issubset(set(payload.keys())):
            missing = sorted(required.difference(set(payload.keys())))
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = build_input_df(payload)
        model = load_model()
        y_log_pred = model.predict(df)
        y_pred = np.expm1(y_log_pred)
        return jsonify({"predicted_price": float(round(y_pred[0], 2))})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Use debug=True only in development
    app.run(host="0.0.0.0", port=5000, debug=True)
