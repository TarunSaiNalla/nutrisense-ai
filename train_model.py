"""
NutriSense AI — Model Training on REAL Kaggle Dataset
Dataset: Obesity Risk Estimation (20,758 real samples)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)

print("Loading real dataset...")
df = pd.read_csv("obesity_level.csv")
print(f"Dataset shape: {df.shape}")

# Map columns to NutriSense feature names
df["gender"]         = (df["Gender"] == "Female").astype(int)
df["age"]            = df["Age"].astype(float)
df["height"]         = df["Height"] * 100
df["weight"]         = df["Weight"].astype(float)
df["activity"]       = df["FAF"].round().clip(0, 3).astype(int)
df["sleep"]          = (7 - df["TUE"] * 0.3).clip(4.5, 10).round(1)
df["water"]          = df["CH2O"].clip(1, 3)

caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
df["junk"]           = (df["FAVC"] * 4 + df["CAEC"].map(caec_map).fillna(1)).clip(0, 7).astype(int)
df["vegetables"]     = ((df["FCVC"] - 1) / 2 * 7).round().clip(0, 7).astype(int)
df["fruits"]         = (df["NCP"].clip(1, 4) / 4 * 5).round().clip(0, 7).astype(int)
df["protein"]        = (df["family_history_with_overweight"] * 2 + 2).clip(0, 7).astype(int)
df["breakfast_skip"] = ((1 - df["SCC"]) * 3).clip(0, 7).astype(int)

calc_map = {"no": 0, "Sometimes": 2, "Frequently": 4, "Always": 6}
df["soda"]           = df["CALC"].map(calc_map).fillna(1).clip(0, 7).astype(int)
df["late_night"]     = df["CAEC"].map({"no": 0, "Sometimes": 1, "Frequently": 3, "Always": 5}).fillna(1).clip(0, 7).astype(int)

# Feature engineering
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["bmi_cat"] = pd.cut(df["bmi"], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(int)
df["healthy_score"] = (
    df["vegetables"] + df["fruits"] + df["protein"] +
    (df["sleep"].clip(6, 9) - 6) * 2 +
    df["water"].clip(0, 3) +
    df["activity"] * 1.5
)
df["unhealthy_score"] = df["junk"] * 1.5 + df["soda"] * 1.2 + df["late_night"] + df["breakfast_skip"]

# Map obesity labels to Low/Moderate/High risk
risk_map = {
    "Insufficient_Weight": 0,
    "0rmal_Weight":        0,
    "Overweight_Level_I":  1,
    "Overweight_Level_II": 1,
    "Obesity_Type_I":      2,
    "Obesity_Type_II":     2,
    "Obesity_Type_III":    2,
}
df["risk"] = df["0be1dad"].map(risk_map)
df = df.dropna(subset=["risk"])
df["risk"] = df["risk"].astype(int)

print(f"\nClass distribution (0=Low, 1=Moderate, 2=High):")
print(df["risk"].value_counts().sort_index())
print(f"Total samples: {len(df)}")

FEATURES = [
    "age", "gender", "weight", "height", "activity",
    "sleep", "water", "junk", "vegetables", "fruits",
    "protein", "breakfast_skip", "soda", "late_night", "bmi",
    "bmi_cat", "healthy_score", "unhealthy_score"
]

X = df[FEATURES]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
}

print("\n── Model Comparison (REAL DATA) ──────────────────────")
best_acc, best_name, best_model = 0, "", None
for name, m in models.items():
    cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring="accuracy")
    m.fit(X_train_sc, y_train)
    test_acc = accuracy_score(y_test, m.predict(X_test_sc))
    print(f"{name:<25} CV: {cv.mean():.3f} ± {cv.std():.3f}  |  Test: {test_acc:.3f}")
    if test_acc > best_acc:
        best_acc, best_name, best_model = test_acc, name, m

print(f"\n✅ Best model: {best_name}  (Test accuracy: {best_acc:.3f})")
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test_sc),
                             target_names=["Low Risk", "Moderate Risk", "High Risk"]))

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler,     open("scaler.pkl", "wb"))
pickle.dump(FEATURES,   open("features.pkl", "wb"))

print(f"\n✅ Saved model.pkl, scaler.pkl, features.pkl")
print(f"✅ Trained on REAL Kaggle dataset — {len(df)} samples")
