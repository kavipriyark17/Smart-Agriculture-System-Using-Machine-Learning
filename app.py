
import json
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# =========================================================
# 1. FLASK CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)


# =========================================================
# 2. GROQ API CONFIGURATION
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CHAT_MODEL = "openai/gpt-oss-120b"


# =========================================================
# 3. LOAD PICKLE MODELS
# =========================================================

def load_pickle(filename):
    file_path = BASE_DIR / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required model file not found: {file_path}"
        )

    with open(file_path, "rb") as file:
        return pickle.load(file)


crop_model = load_pickle("crop_model.pkl")
fertilizer_model = load_pickle("fertilizer_model.pkl")

soil_encoder = load_pickle("soil_encoder.pkl")
crop_encoder = load_pickle("crop_encoder.pkl")
fert_encoder = load_pickle("fert_encoder.pkl")


# =========================================================
# 4. LOAD DISEASE MODEL AND EXACT CLASS LABELS
# =========================================================

disease_model_path = BASE_DIR / "disease_model.h5"
disease_labels_path = BASE_DIR / "disease_labels.json"

if not disease_model_path.is_file():
    raise FileNotFoundError(
        f"Disease model not found: {disease_model_path}"
    )

if not disease_labels_path.is_file():
    raise FileNotFoundError(
        f"Disease labels not found: {disease_labels_path}"
    )

disease_model = tf.keras.models.load_model(
    str(disease_model_path),
    compile=False
)

with open(disease_labels_path, "r", encoding="utf-8") as file:
    DISEASE_LABELS = json.load(file)

if not isinstance(DISEASE_LABELS, list):
    raise ValueError("Disease labels must be a JSON list.")

if not all(
    isinstance(label, str) and label.strip()
    for label in DISEASE_LABELS
):
    raise ValueError("Disease labels contain invalid entries.")

if len(DISEASE_LABELS) != int(disease_model.output_shape[-1]):
    raise ValueError(
        f"Model has {disease_model.output_shape[-1]} output classes, "
        f"but labels file has {len(DISEASE_LABELS)} labels."
    )

print(
    f"Disease model loaded successfully: "
    f"{len(DISEASE_LABELS)} classes."
)


# =========================================================
# 5. SHARED INPUT VALIDATION
# =========================================================

def read_numeric_features(data, required_fields):
    """Validate and return numeric input values in the given order."""

    missing = [
        field for field in required_fields
        if field not in data
    ]

    if missing:
        raise ValueError(
            "Missing fields: " + ", ".join(missing)
        )

    values = []

    for field in required_fields:
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            raise ValueError(
                f"'{field}' must be a valid number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"'{field}' must be a finite number."
            )

        values.append(value)

    return values


# =========================================================
# 6. PAGE ROUTES
# =========================================================

@app.route("/")
@app.route("/home")
@app.route("/home.html")
def home():
    return render_template("home.html")


@app.route("/predict-page")
@app.route("/predict.html")
def predict_page():
    return render_template("predict.html")


@app.route("/disease")
@app.route("/disease.html")
def disease_page():
    return render_template("disease.html")


@app.route("/fertilizer")
@app.route("/fertilizer.html")
def fertilizer_page():
    return render_template("fertilizer.html")


# =========================================================
# 7. CROP RECOMMENDATION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send valid JSON input."
            }), 400

        required = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

        values = read_numeric_features(data, required)

        features = np.array(
            [values],
            dtype=np.float64
        )

        prediction = crop_model.predict(features)

        if len(prediction) == 0:
            raise ValueError("Crop model returned no prediction.")

        crop_name = str(
            np.asarray(prediction).reshape(-1)[0]
        )

        return jsonify({
            "crop": crop_name
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Crop prediction failed")

        return jsonify({
            "error": (
                "Crop prediction failed. "
                "Check the terminal logs."
            )
        }), 500


# =========================================================
# 8. PLANT DISEASE DETECTION — 15 CLASSES
# =========================================================

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        uploaded_file = request.files.get("file")

        if (
            uploaded_file is None
            or not uploaded_file.filename
        ):
            return jsonify({
                "error": "Please upload a plant leaf image."
            }), 400

        try:
            with Image.open(uploaded_file.stream) as original:
                image = original.convert("RGB")
                image = image.resize((160, 160))

                # The trained MobileNetV2 model contains its own
                # Rescaling layer: pixel values must enter as 0–255.
                image_array = np.asarray(
                    image,
                    dtype=np.float32
                )

        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({
                "error": "The uploaded file is not a valid image."
            }), 400

        image_array = np.expand_dims(image_array, axis=0)

        prediction = disease_model.predict(
            image_array,
            verbose=0
        )

        scores = np.asarray(prediction)

        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError(
                "Unexpected disease model output shape."
            )

        if scores.shape[1] != len(DISEASE_LABELS):
            raise ValueError(
                "Disease model output does not match label count."
            )

        if not np.isfinite(scores).all():
            raise ValueError(
                "Disease model returned invalid prediction scores."
            )

        predicted_index = int(np.argmax(scores[0]))
        confidence = float(scores[0][predicted_index]) * 100.0

        return jsonify({
            "disease": DISEASE_LABELS[predicted_index],
            "confidence": round(confidence, 2)
        })

    except Exception:
        app.logger.exception("Disease detection failed")

        return jsonify({
            "error": (
                "Disease detection failed. "
                "Check model configuration and terminal logs."
            )
        }), 500


# =========================================================
# 9. FERTILIZER RECOMMENDATION
# =========================================================

@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send valid JSON input."
            }), 400

        required = [
            "temperature",
            "humidity",
            "moisture",
            "soil",
            "crop",
            "nitrogen",
            "potassium",
            "phosphorous"
        ]

        missing = [
            field for field in required
            if field not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing fields: " + ", ".join(missing)
            }), 400

        soil_name = str(data["soil"]).strip()
        crop_name = str(data["crop"]).strip()

        if not soil_name or not crop_name:
            return jsonify({
                "error": "Soil and crop names are required."
            }), 400

        numeric_fields = [
            "temperature",
            "humidity",
            "moisture",
            "nitrogen",
            "potassium",
            "phosphorous"
        ]

        numeric_values = read_numeric_features(
            data,
            numeric_fields
        )

        # Encoders must receive category names they saw during training.
        soil_encoded = soil_encoder.transform([soil_name])[0]
        crop_encoded = crop_encoder.transform([crop_name])[0]

        # Keep the feature order used in fertilizer_model.py.
        temperature, humidity, moisture, nitrogen, potassium, phosphorous = (
            numeric_values
        )

        features = np.array([[
            temperature,
            humidity,
            moisture,
            float(soil_encoded),
            float(crop_encoded),
            nitrogen,
            potassium,
            phosphorous
        ]], dtype=np.float64)

        prediction = fertilizer_model.predict(features)

        fertilizer_name = fert_encoder.inverse_transform(
            np.asarray(prediction).reshape(-1)
        )[0]

        return jsonify({
            "fertilizer": str(fertilizer_name)
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": (
                "Invalid input. Check numeric values, soil name "
                "and crop name. Details: " + str(error)
            )
        }), 400

    except Exception:
        app.logger.exception("Fertilizer prediction failed")

        return jsonify({
            "error": (
                "Fertilizer prediction failed. "
                "Check model, encoder and feature order."
            )
        }), 500


# =========================================================
# 10. GROQ AGRICULTURE CHATBOT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({
            "error": (
                "Groq API key is missing. Set GROQ_API_KEY "
                "in PowerShell and restart Flask."
            )
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send valid JSON."
        }), 400

    user_message = data.get("message")

    if (
        not isinstance(user_message, str)
        or not user_message.strip()
    ):
        return jsonify({
            "error": "Please enter a message."
        }), 400

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are AgriBot AI, an agriculture assistant. "
                            "Answer questions about crop selection, plant "
                            "diseases, fertilizer, irrigation, soil health "
                            "and farming practices. Give clear, practical "
                            "answers. Do not claim live weather or sensor "
                            "data unless the user provides it. For serious "
                            "crop disease or pesticide decisions, advise "
                            "consulting a local agricultural expert."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_message.strip()
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 800
            },
            timeout=(10, 45)
        )

        if not response.ok:
            # Never log or return the API key.
            app.logger.error(
                "Groq HTTP %s: %s",
                response.status_code,
                response.text[:1000]
            )

            if response.status_code == 401:
                return jsonify({
                    "error": "Groq API key is invalid. Check your key."
                }), 502

            if response.status_code == 429:
                return jsonify({
                    "error": (
                        "Groq rate or usage limit reached. "
                        "Check your Groq Console limits."
                    )
                }), 429

            if response.status_code == 404:
                return jsonify({
                    "error": (
                        "Groq endpoint or model was not found. "
                        "Check the model ID in your Groq Console "
                        "and update CHAT_MODEL."
                    )
                }), 502

            if response.status_code == 400:
                return jsonify({
                    "error": (
                        "Groq rejected the request. Check the model "
                        "ID and request settings in the terminal log."
                    )
                }), 502

            return jsonify({
                "error": (
                    f"Groq API request failed (HTTP "
                    f"{response.status_code}). Check terminal logs."
                )
            }), 502

        result = response.json()
        choices = result.get("choices", [])

        if not choices:
            return jsonify({
                "error": "Groq returned no response. Please try again."
            }), 502

        reply = (
            choices[0]
            .get("message", {})
            .get("content")
        )

        if not isinstance(reply, str) or not reply.strip():
            return jsonify({
                "error": "Groq returned an empty response."
            }), 502

        return jsonify({
            "reply": reply.strip()
        })

    except requests.Timeout:
        app.logger.warning("Groq request timed out.")

        return jsonify({
            "error": "AgriBot timed out. Please try again."
        }), 504

    except requests.RequestException:
        app.logger.exception("Unable to connect to Groq.")

        return jsonify({
            "error": (
                "Unable to connect to Groq. "
                "Check your internet connection and try again."
            )
        }), 502

    except Exception:
        app.logger.exception("Unexpected AgriBot error.")

        return jsonify({
            "error": "AgriBot encountered an error. Check terminal logs."
        }), 500


# =========================================================
# 11. FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Uploaded image cannot exceed 10 MB."
    }), 413


# =========================================================
# 12. RUN FLASK
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
