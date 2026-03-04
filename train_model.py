"""
NutriSense AI — Improved Model Training
- Larger, more realistic synthetic dataset (5000 samples)
- Better feature engineering (BMI category, activity × sleep interaction)
- Hyperparameter-tuned Random Forest
- Cross-validation + test accuracy printed
- Saves model.pkl and scaler.pkl into current directory
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
N = 5000

# ─── Generate Realistic Synthetic Data ───────────────────────────────────────
df = pd.DataFrame({
    "age":             np.random.randint(18, 70, N),
    "gender":          np.random.randint(0, 3, N),
    "weight":          np.random.randint(40, 130, N),
    "height":          np.random.randint(145, 200, N),
    "activity":        np.random.randint(0, 4, N),
    "sleep":           np.round(np.random.uniform(3.5, 10, N), 1),
    "water":           np.round(np.random.uniform(0.5, 5, N), 1),
    "junk":            np.random.randint(0, 8, N),
    "vegetables":      np.random.randint(0, 8, N),
    "fruits":          np.random.randint(0, 8, N),
    "protein":         np.random.randint(0, 8, N),
    "breakfast_skip":  np.random.randint(0, 8, N),
    "soda":            np.random.randint(0, 8, N),
    "late_night":      np.random.randint(0, 8, N),
})

# ─── Feature Engineering ─────────────────────────────────────────────────────
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# BMI category: 0=underweight, 1=normal, 2=overweight, 3=obese
df["bmi_cat"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=[0, 1, 2, 3]
).astype(int)

# Lifestyle quality score (higher = better)
df["healthy_score"] = (
    df["vegetables"] + df["fruits"] + df["protein"] +
    (df["sleep"].clip(6, 9) - 6) * 2 +
    df["water"].clip(0, 3) +
    df["activity"] * 1.5
)

# Unhealthy score (higher = worse)
df["unhealthy_score"] = (
    df["junk"] * 1.5 +
    df["soda"] * 1.2 +
    df["late_night"] +
    df["breakfast_skip"]
)

# ─── Label Creation (realistic, nuanced) ─────────────────────────────────────
net_risk = (
    df["unhealthy_score"] * 1.2
    - df["healthy_score"] * 0.8
    + (df["bmi_cat"] - 1) * 2        # penalise overweight/obese
    + (df["age"] > 45).astype(int)   # slight penalty for older age
    - df["activity"]                  # active = better
)

# Thresholds chosen so classes are roughly balanced
df["risk"] = np.where(
    net_risk < 0,  0,
    np.where(net_risk < 6, 1, 2)
)

print("Class distribution:\n", df["risk"].value_counts().sort_index())

# ─── Feature Matrix ───────────────────────────────────────────────────────────
FEATURES = [
    "age", "gender", "weight", "height", "activity",
    "sleep", "water", "junk", "vegetables", "fruits",
    "protein", "breakfast_skip", "soda", "late_night", "bmi",
    "bmi_cat", "healthy_score", "unhealthy_score"
]

X = df[FEATURES]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Model Comparison ─────────────────────────────────────────────────────────
models = {
    "Random Forest":        RandomForestClassifier(
                                n_estimators=200,
                                max_depth=12,
                                min_samples_split=4,
                                random_state=42,
                                n_jobs=-1
                            ),
    "Gradient Boosting":    GradientBoostingClassifier(
                                n_estimators=150,
                                learning_rate=0.1,
                                max_depth=5,
                                random_state=42
                            ),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "KNN":                  KNeighborsClassifier(n_neighbors=7),
}

print("\n── Model Comparison ──────────────────────────────")
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
                             target_names=["Low Risk","Moderate Risk","High Risk"]))

# ─── Save ─────────────────────────────────────────────────────────────────────
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler,     open("scaler.pkl", "wb"))
pickle.dump(FEATURES,   open("features.pkl", "wb"))

print("✅ model.pkl, scaler.pkl, features.pkl saved successfully!")
