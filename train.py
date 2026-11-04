
# ============================================================
# INDIAN RAINFALL PREDICTION
# Multiple Machine Learning Models
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/rainfall_districtwise_daily_imd.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Convert Date column
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Rename rainfall column
df.rename(
    columns={"Daily Actual": "Rainfall"},
    inplace=True
)

# Remove missing dates and rainfall values
df.dropna(
    subset=["Date", "Rainfall"],
    inplace=True
)

# Remove negative rainfall values
df = df[df["Rainfall"] >= 0]

# Remove duplicate district-date observations
df.drop_duplicates(
    subset=["State", "District", "Date"],
    inplace=True
)

print("Cleaned dataset shape:", df.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Extract temporal features
df["Month"] = df["Date"].dt.month

df["DayOfYear"] = df["Date"].dt.dayofyear

# Cyclical encoding for seasonal patterns
df["Month_sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

df["DayOfYear_sin"] = np.sin(
    2 * np.pi * df["DayOfYear"] / 365.25
)

df["DayOfYear_cos"] = np.cos(
    2 * np.pi * df["DayOfYear"] / 365.25
)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "State",
    "District",
    "Month_sin",
    "Month_cos",
    "DayOfYear_sin",
    "DayOfYear_cos"
]

target = "Rainfall"

X = df[features]

y = df[target]


# ============================================================
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

# Sort observations chronologically
df = df.sort_values("Date")

# Use the earliest 80% for training
# and the latest 20% for testing

split_index = int(len(df) * 0.80)

X_train = X.loc[df.index[:split_index]]
X_test = X.loc[df.index[split_index:]]

y_train = y.loc[df.index[:split_index]]
y_test = y.loc[df.index[split_index:]]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

categorical_features = [
    "State",
    "District"
]

numerical_features = [
    "Month_sin",
    "Month_cos",
    "DayOfYear_sin",
    "DayOfYear_cos"
]

# Encode categorical features
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])

# Handle numerical features
numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer([
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    ),
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    )
])


# ============================================================
# 6. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

}


# ============================================================
# 7. TRAIN AND EVALUATE ALL MODELS
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"TRAINING: {name}")
    print("=" * 50)

    # Create complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train model
    pipeline.fit(
        X_train,
        y_train
    )

    # Generate predictions
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Pipeline": pipeline
    }

    # Display results
    print(f"MAE      : {mae:.4f} mm")
    print(f"RMSE     : {rmse:.4f} mm")
    print(f"R2 Score : {r2:.4f}")


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "Model": name,
        "MAE": values["MAE"],
        "RMSE": values["RMSE"],
        "R2 Score": values["R2 Score"]
    }

    for name, values in results.items()
])

# Sort models by lowest RMSE
comparison = comparison.sort_values(
    by="RMSE"
).reset_index(drop=True)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))


# ============================================================
# 9. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 10. SAVE ALL TRAINED MODELS
# ============================================================

model_filenames = {

    "Linear Regression":
        "linear_regression.pkl",

    "Random Forest":
        "random_forest.pkl",

    "XGBoost":
        "xgboost.pkl"

}

print("\n" + "=" * 60)
print("SAVING TRAINED MODELS")
print("=" * 60)

for name, values in results.items():

    filename = model_filenames[name]

    model_pipeline = values["Pipeline"]

    save_path = os.path.join(
        "models",
        filename
    )

    joblib.dump(
        model_pipeline,
        save_path
    )

    print(f"{name} saved: {save_path}")


# ============================================================
# 11. IDENTIFY AND SAVE BEST MODEL
# ============================================================

best_model_name = comparison.iloc[0]["Model"]

best_model_pipeline = results[
    best_model_name
]["Pipeline"]

joblib.dump(
    best_model_pipeline,
    "models/best_rainfall_model.pkl"
)

print("\nBest performing model:", best_model_name)

print(
    "Best model saved:",
    "models/best_rainfall_model.pkl"
)


# ============================================================
# 12. SAVE MODEL PERFORMANCE COMPARISON
# ============================================================

comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved:",
    "models/model_comparison.csv"
)


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModels available:")

for filename in os.listdir("models"):

    print(" -", filename)

print("\nYou can now use these trained models in app.py")
