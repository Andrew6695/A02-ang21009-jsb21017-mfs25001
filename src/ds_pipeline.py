from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)

# Check
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# train MLPRegressor
mlp = MLPRegressor(
    random_state=42,
    hidden_layer_sizes=(64, 32), # retuning capacity
    alpha=1e-4, # regularization
    learning_rate_init=1e-3, # step size
    max_iter=1000, # allowing for more training
    batch_size=256,
    activation="relu",
    validation_fraction=0.2,
    early_stopping=True
)
mlp.fit(X_train_scaled, y_train)

# predict
y_pred_train = mlp.predict(X_train_scaled)
y_pred_test  = mlp.predict(X_test_scaled)

# metrics
print("train R2:", r2_score(y_train, y_pred_train))
print("train MAE:", mean_absolute_error(y_train, y_pred_train))
print("test  R2:", r2_score(y_test, y_pred_test))
print("test  MAE:", mean_absolute_error(y_test, y_pred_test))

# scatter + reference line
import os
os.makedirs("figures", exist_ok=True)

def scatter_with_reference_save(y_true, y_pred, title, out_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1, color="red")
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

scatter_with_reference_save(
    y_train, y_pred_train,
    "Actual vs Predicted - Train",
    "figures/train_actual_vs_pred.png"
)

scatter_with_reference_save(
    y_test, y_pred_test,
    "Actual vs Predicted - Test",
    "figures/test_actual_vs_pred.png"
)

print("Saved plots to figures/ folder.")
