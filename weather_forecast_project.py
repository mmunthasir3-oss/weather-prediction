"""
=====================================================================
 WEATHER FORECASTING PREDICTION SYSTEM
=====================================================================
Dataset : weatherHistory.csv  (Kaggle "Weather in Szeged 2006-2016")
Goal    : Predict Temperature (C) using other weather measurements
          like Humidity, Wind Speed, Pressure, Visibility, etc.
Model   : Random Forest Regressor (also compares with Linear Regression)

WORKS ON BOTH:
  1. Google Colab
  2. VS Code (local machine)

The script auto-detects where it is running and loads the CSV file
from the correct location. See the "STEP 1" section below.
=====================================================================
"""

# ---------------------------------------------------------------
# STEP 0: IMPORT LIBRARIES
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------
# STEP 1: LOAD THE DATASET (works for Colab AND VS Code)
# ---------------------------------------------------------------
import os

def load_dataset():
    """
    Tries a few common file locations so the same script works
    whether you are in Google Colab or VS Code.
    """
    possible_paths = [
        "weatherHistory.csv",                     # same folder as script (VS Code)
        "/content/weatherHistory.csv",             # Colab after upload
        "/content/drive/MyDrive/weatherHistory.csv"  # Colab + Google Drive
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            return pd.read_csv(path)

    # If running in Colab and file not found, prompt upload
    try:
        import google.colab  # only exists in Colab
        from google.colab import files
        print("Dataset not found. Please upload 'weatherHistory.csv' now...")
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        return pd.read_csv(filename)
    except ImportError:
        raise FileNotFoundError(
            "weatherHistory.csv not found. Place it in the same folder as this script."
        )

df = load_dataset()
print("\nDataset shape:", df.shape)
print(df.head())


# ---------------------------------------------------------------
# STEP 2: CLEAN THE DATA
# ---------------------------------------------------------------
# Convert date column to real datetime type
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)

# Extract useful time-based features
df["Month"] = df["Formatted Date"].dt.month
df["Hour"] = df["Formatted Date"].dt.hour

# "Precip Type" has some missing values -> fill with the most common value
df["Precip Type"] = df["Precip Type"].fillna(df["Precip Type"].mode()[0])

# Convert text column "Precip Type" (rain/snow) into numbers
df["Precip Type"] = df["Precip Type"].map({"rain": 0, "snow": 1})

# Drop columns we don't need for prediction
df_model = df.drop(columns=[
    "Formatted Date", "Summary", "Daily Summary", "Loud Cover"
])

print("\nCleaned data sample:")
print(df_model.head())


# ---------------------------------------------------------------
# STEP 3: SELECT FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------------
# We are predicting Temperature (C)
X = df_model.drop(columns=["Temperature (C)"])
y = df_model["Temperature (C)"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features so all numbers are on a similar range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------
# STEP 4: TRAIN MODELS
# ---------------------------------------------------------------
print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)   # Random Forest doesn't need scaled data
rf_preds = rf_model.predict(X_test)


# ---------------------------------------------------------------
# STEP 5: EVALUATE THE MODELS
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"MAE  (Mean Absolute Error) : {mae:.3f} °C")
    print(f"RMSE (Root Mean Sq Error)  : {rmse:.3f} °C")
    print(f"R2 Score (accuracy-like)   : {r2:.3f}")
    return mae, rmse, r2

evaluate("Linear Regression", y_test, lr_preds)
evaluate("Random Forest", y_test, rf_preds)


# ---------------------------------------------------------------
# STEP 6: VISUALIZE RESULTS
# ---------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, alpha=0.3, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linewidth=2)
plt.xlabel("Actual Temperature (C)")
plt.ylabel("Predicted Temperature (C)")
plt.title("Random Forest: Actual vs Predicted Temperature")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# Feature importance chart (which factors matter most)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="steelblue")
plt.title("Which Features Matter Most for Predicting Temperature?")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# ---------------------------------------------------------------
# STEP 7: PREDICT ON A NEW / CUSTOM WEATHER READING
# ---------------------------------------------------------------
# Example: Suppose you measure these values on a new day
sample_input = pd.DataFrame([{
    "Precip Type": 0,             # 0 = rain, 1 = snow
    "Apparent Temperature (C)": 10.0,
    "Humidity": 0.80,
    "Wind Speed (km/h)": 12.0,
    "Wind Bearing (degrees)": 200,
    "Visibility (km)": 10.0,
    "Pressure (millibars)": 1015.0,
    "Month": 7,
    "Hour": 14
}])

predicted_temp = rf_model.predict(sample_input)[0]
print(f"\nPredicted Temperature for the sample input: {predicted_temp:.2f} °C")

print("\nDone! Check 'actual_vs_predicted.png' and 'feature_importance.png' for charts.")
