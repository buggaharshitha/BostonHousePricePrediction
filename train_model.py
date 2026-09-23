"""
train_model.py
--------------
Trains a Random Forest Regressor on the Boston Housing dataset,
evaluates it, saves the model artifact, and generates output charts.
"""
# coding: utf-8

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "boston_housing.csv")
OUTPUT_DIR  = "outputs"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
df["chas"] = df["chas"].astype(str).str.strip('"').astype(int)

X = df.drop(columns=["medv"])
y = df["medv"]
feature_names = X.columns.tolist()

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train model ──────────────────────────────────────────────────────────────
print("Training Random Forest Regressor ...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="r2").mean()

print(f"\n-- Evaluation Results ------------------")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")
print(f"  CV R²: {cv_r2:.4f}")

# ── Save model & scaler ──────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\nModel  saved -> {MODEL_PATH}")
print(f"Scaler saved -> {SCALER_PATH}")

# ── Chart 1: Actual vs Predicted ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4, color="#3b82d4")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5)
ax.set_xlabel("Actual Price ($1000s)")
ax.set_ylabel("Predicted Price ($1000s)")
ax.set_title("Actual vs Predicted Housing Prices")
ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

# ── Chart 2: Feature Importance ──────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
importances.plot(kind="barh", ax=ax, color="#3b82d4")
ax.set_title("Feature Importances")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ── Chart 3: Residuals ───────────────────────────────────────────────────────
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.4, color="#7c5cd8")
ax.axhline(0, color="red", lw=1.5, linestyle="--")
ax.set_xlabel("Predicted Price ($1000s)")
ax.set_ylabel("Residual")
ax.set_title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals.png"), dpi=150)
plt.close()

# ── Chart 4: Correlation heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

# ── Risk table ───────────────────────────────────────────────────────────────
risk_df = pd.DataFrame({
    "Actual":    y_test.values,
    "Predicted": y_pred,
    "Error":     residuals.values,
    "AbsError":  np.abs(residuals.values),
})
risk_df["Risk"] = pd.cut(
    risk_df["AbsError"],
    bins=[0, 2, 5, np.inf],
    labels=["Low", "Medium", "High"]
)
risk_df.to_csv(os.path.join(OUTPUT_DIR, "risk_table.csv"), index=False)
print(f"\nRisk table saved -> {os.path.join(OUTPUT_DIR, 'risk_table.csv')}")

print("\nTraining complete. All outputs saved to outputs/")
