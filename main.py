# simple_linear_regression_advertising.py
"""
Simple Linear Regression (Sales vs Advertising)
- Loads dataset if file 'advertising.csv' exists (expects columns: 'Advertising' and 'Sales')
- Otherwise generates a small synthetic dataset
- Trains LinearRegression from scikit-learn
- Makes predictions and prints evaluation metrics
- Plots data + fitted line and residuals
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# ---------------------------
# 1. Load or create dataset
# ---------------------------
CSV_PATH = "advertising.csv"  # change if your file has a different name

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    # Ensure expected columns exist
    if not {"Advertising", "Sales"}.issubset(df.columns):
        raise ValueError("CSV found but it must contain columns named 'Advertising' and 'Sales'.")
    print(f"Loaded dataset from {CSV_PATH}. Shape: {df.shape}")
else:
    # Create a small synthetic dataset for demonstration
    rng = np.random.RandomState(42)
    advertising = np.linspace(1, 100, 80)  # advertising spend (e.g., in thousands)
    # True relationship (for synthetic example) with some noise
    sales = 2.5 + 0.08 * advertising + rng.normal(scale=3.5, size=advertising.shape)
    df = pd.DataFrame({"Advertising": advertising, "Sales": sales})
    print("No CSV found — using generated synthetic dataset. (You can save your real data as 'advertising.csv')")

# Quick look
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------
# 2. Exploratory plot
# ---------------------------
plt.figure(figsize=(7, 4))
plt.scatter(df["Advertising"], df["Sales"], alpha=0.7)
plt.xlabel("Advertising (units)")
plt.ylabel("Sales (units)")
plt.title("Sales vs Advertising (scatter)")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# 3. Prepare X and y
# ---------------------------
X = df[["Advertising"]].values  # must be 2D for scikit-learn
y = df["Sales"].values          # 1D array

# ---------------------------
# 4. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------
# 5. Train the Linear Regression model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
intercept = model.intercept_
coef = model.coef_[0]
print("\nTrained Linear Regression model:")
print(f"  Intercept (b0): {intercept:.4f}")
print(f"  Coefficient (b1): {coef:.4f}")

# ---------------------------
# 6. Make predictions
# ---------------------------
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ---------------------------
# 7. Evaluate the model
# ---------------------------
def regression_metrics(y_true, y_pred, label="Test"):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f"\n{label} set evaluation:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R^2:  {r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

metrics_train = regression_metrics(y_train, y_pred_train, label="Train")
metrics_test = regression_metrics(y_test, y_pred_test, label="Test")

# ---------------------------
# 8. Plot regression line on full data
# ---------------------------
plt.figure(figsize=(7, 4))
plt.scatter(X, y, alpha=0.6, label="Data points")
# regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, linewidth=2, label=f"Fit: Sales = {intercept:.2f} + {coef:.4f}*Advertising")
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.title("Linear Regression: Sales vs Advertising")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# 9. Residuals plot (on test set)
# ---------------------------
residuals = y_test - y_pred_test
plt.figure(figsize=(7, 4))
plt.scatter(y_pred_test, residuals, alpha=0.7)
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot (Test set)")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# 10. Example prediction(s)
# ---------------------------
example_ad_spends = np.array([[10.0], [50.0], [90.0]])  # change to values you want to predict for
preds = model.predict(example_ad_spends)
print("\nExample predictions:")
for ad, p in zip(example_ad_spends.flatten(), preds):
    print(f"  Advertising={ad:.1f} -> Predicted Sales={p:.3f}")

# Optionally: save the model (commented out)
# from joblib import dump
# dump(model, "linear_regression_advertising.joblib")
