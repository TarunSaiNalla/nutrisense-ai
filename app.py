from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
from datetime import datetime
from collections import Counter

def safe_int(val):
    try:
        return int(float(str(val).strip()))
    except:
        return 0

def safe_float(val):
    try:
        return float(str(val).strip())
    except:
        return 0.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

model  = load_pkl(os.path.join(BASE_DIR, "model.pkl"))
scaler = load_pkl(os.path.join(BASE_DIR, "scaler.pkl"))
try:
    FEATURES = load_pkl(os.path.join(BASE_DIR, "features.pkl"))
except FileNotFoundError:
    FEATURES = [
        "age","gender","weight","height","activity",
        "sleep","water","junk","vegetables","fruits",
        "protein","breakfast_skip","soda","late_night","bmi"
    ]

_history = []

GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
ACTIVITY_MAP = {
    "Sedentary (desk job)": 0,
    "Lightly Active":        1,
    "Moderately Active":     2,
    "Very Active":           3,
}
RISK_MAP   = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
RISK_EMOJI = {0: "💚", 1: "🟡", 2: "🔴"}


def build_recommendations(data, risk):
    tips = []
    bmi      = safe_float(data.get("bmi", 22))
    sleep    = safe_float(data.get("sleep", 7))
    water    = safe_float(data.get("water", 2))
    junk     = safe_int(data.get("junk", 0))
    soda     = safe_int(data.get("soda", 0))
    vegs     = safe_int(data.get("vegetables", 0))
    fruits   = safe_int(data.get("fruits", 0))
    protein  = safe_int(data.get("protein", 0))
    bskip    = safe_int(data.get("breakfast_skip", 0))
    late     = safe_int(data.get("late_night", 0))
    activity = ACTIVITY_MAP.get(str(data.get("activity", "")), 1)

    if bmi > 30:
        tips.append("🏃 Your BMI suggests obesity. Aim for 30 min of moderate exercise daily.")
    elif bmi > 25:
        tips.append("⚖️ You are slightly overweight. Consider reducing caloric intake by ~300 kcal/day.")
    elif bmi < 18.5:
        tips.append("🥩 Your BMI is low. Focus on nutrient-dense foods — whole grains, legumes, dairy.")
    if sleep < 6:
        tips.append("😴 You sleep less than 6 hours. Poor sleep raises cortisol and increases cravings.")
    elif sleep > 9:
        tips.append("🛌 Sleeping over 9 hours may signal poor sleep quality — keep a consistent schedule.")
    if water < 2:
        tips.append("💧 Drink at least 2–2.5 L of water daily. Dehydration slows metabolism.")
    if junk >= 4:
        tips.append("🍔 High junk food frequency. Limit ultra-processed food to once a week.")
    if soda >= 3:
        tips.append("🥤 Cut down on sugary drinks — each can adds ~150 empty calories.")
    if vegs < 3:
        tips.append("🥦 Eat at least 3 servings of vegetables daily for fibre and vitamins.")
    if fruits < 2:
        tips.append("🍎 Add 2 fruit servings per day for natural sugars and micronutrients.")
    if protein < 3:
        tips.append("💪 Increase protein intake (eggs, legumes, chicken, tofu) to maintain muscle.")
    if bskip >= 4:
        tips.append("🍳 Skipping breakfast disrupts metabolism. Eat a protein-rich breakfast daily.")
    if activity == 0:
        tips.append("🚶 Sedentary lifestyle is a major risk factor. Start with 10-minute daily walks.")
    if late >= 4:
        tips.append("🌙 Late-night eating disrupts circadian rhythm. Stop eating 2h before bedtime.")
    if risk == 2 and not tips:
        tips.append("⚠️ Multiple lifestyle factors contributing to high risk — consult a nutritionist.")
    if risk == 0:
        tips.append("🌟 Great job! Keep up your healthy habits. Regular check-ups help maintain this level.")
    return tips[:5]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/bmi", methods=["POST"])
def calc_bmi():
    try:
        d = request.get_json()
        weight = safe_float(d["weight"])
        height = safe_float(d["height"])
        bmi = round(weight / ((height / 100) ** 2), 1)
        category = (
            "Underweight" if bmi < 18.5 else
            "Normal"      if bmi < 25   else
            "Overweight"  if bmi < 30   else "Obese"
        )
        return jsonify({"bmi": bmi, "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        weight  = safe_float(data.get("weight", 0))
        height  = safe_float(data.get("height", 0))
        sleep   = safe_float(data.get("sleep", 7))
        water   = safe_float(data.get("water", 2))
        age     = safe_int(data.get("age", 0))
        junk    = safe_int(data.get("junk", 0))
        vegs    = safe_int(data.get("vegetables", 0))
        fruits  = safe_int(data.get("fruits", 0))
        protein = safe_int(data.get("protein", 0))
        bskip   = safe_int(data.get("breakfast_skip", 0))
        soda    = safe_int(data.get("soda", 0))
        late    = safe_int(data.get("late_night", 0))
        gender  = GENDER_MAP.get(str(data.get("gender", "Male")), 0)

        # activity is ALWAYS a string — NEVER call int() on it directly
        activity_str = str(data.get("activity", "Lightly Active")).strip()
        activity = ACTIVITY_MAP.get(activity_str, 1)

        bmi = round(weight / ((height / 100) ** 2), 1) if height > 0 else 22.0

        bmi_cat = (
            0 if bmi < 18.5 else
            1 if bmi < 25   else
            2 if bmi < 30   else 3
        )

        healthy_score = (
            vegs + fruits + protein +
            (min(max(sleep, 6), 9) - 6) * 2 +
            min(water, 3) +
            activity * 1.5
        )
        unhealthy_score = junk * 1.5 + soda * 1.2 + late + bskip

        raw = {
            "age":             age,
            "gender":          gender,
            "weight":          weight,
            "height":          height,
            "activity":        activity,
            "sleep":           sleep,
            "water":           water,
            "junk":            junk,
            "vegetables":      vegs,
            "fruits":          fruits,
            "protein":         protein,
            "breakfast_skip":  bskip,
            "soda":            soda,
            "late_night":      late,
            "bmi":             bmi,
            "bmi_cat":         bmi_cat,
            "healthy_score":   healthy_score,
            "unhealthy_score": unhealthy_score,
        }

        feat_vec = np.array([raw[f] for f in FEATURES]).reshape(1, -1)
        feat_sc  = scaler.transform(feat_vec)

        prediction    = int(model.predict(feat_sc)[0])
        probabilities = model.predict_proba(feat_sc)[0]
        confidence    = round(float(max(probabilities)) * 100, 2)

        bmi_category = (
            "Underweight" if bmi < 18.5 else
            "Normal"      if bmi < 25   else
            "Overweight"  if bmi < 30   else "Obese"
        )

        data["bmi"]      = bmi
        data["activity"] = activity_str
        tips = build_recommendations(data, prediction)

        response = {
            "risk":          RISK_MAP[prediction],
            "emoji":         RISK_EMOJI[prediction],
            "confidence":    confidence,
            "bmi":           bmi,
            "bmi_category":  bmi_category,
            "probabilities": {
                "low":      round(float(probabilities[0]) * 100, 2),
                "moderate": round(float(probabilities[1]) * 100, 2),
                "high":     round(float(probabilities[2]) * 100, 2),
            },
            "recommendations": tips,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        _history.append({
            "timestamp":  response["timestamp"],
            "risk":       response["risk"],
            "confidence": confidence,
            "bmi":        bmi,
            "age":        age,
        })

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            "error":   "Prediction failed",
            "details": str(e),
            "trace":   traceback.format_exc()
        }), 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify({"count": len(_history), "history": _history[-20:][::-1]})


@app.route("/stats", methods=["GET"])
def stats():
    if not _history:
        return jsonify({"message": "No predictions yet."})
    counts   = Counter(h["risk"] for h in _history)
    avg_conf = round(sum(h["confidence"] for h in _history) / len(_history), 2)
    avg_bmi  = round(sum(h["bmi"]        for h in _history) / len(_history), 1)
    return jsonify({
        "total_predictions":  len(_history),
        "risk_distribution":  dict(counts),
        "average_confidence": avg_conf,
        "average_bmi":        avg_bmi,
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":         "healthy",
        "model_loaded":   True,
        "model_type":     type(model).__name__,
        "feature_count":  len(FEATURES),
        "total_analyzed": len(_history),
        "timestamp":      datetime.now().isoformat(),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
