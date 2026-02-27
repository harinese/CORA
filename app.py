from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("final_model.pkl")

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Route for frontend page
@app.route("/")
def home():
    return render_template("index.html")

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "send { 'features': { ... } }"}), 400

    feats = data["features"]
    try:
        X = pd.DataFrame([{k: feats[k] for k in FEATURE_ORDER}])
    except KeyError as e:
        return jsonify({"error": f"missing feature {str(e)}"}), 400

    # prediction
    pred = int(model.predict(X)[0])

    # probability: try predict_proba, fallback to decision_function -> sigmoid, else None
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            prob = None
    elif hasattr(model, "decision_function"):
        try:
            score = float(model.decision_function(X)[0])
            prob = float(1 / (1 + np.exp(-score)))
        except Exception:
            prob = None

    return jsonify({
        "prediction": pred,
        "probability": prob,
        "model_version": "v1"
    })

if __name__ == "__main__":
    app.run(debug=True)