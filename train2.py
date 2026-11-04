
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ==========================================
# 1. LOAD PROCESSED DATA
# ==========================================

print("Loading processed rainfall dataset...")

df = pd.read_csv("data/rainfall_processed.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. FEATURE SELECTION
# ==========================================

features = [
    "LATITUDE",
    "LONGITUDE",
    "Month",
    "DayOfYear",
    "Day_sin",
    "Day_cos"
]

target = "Rainfall"

X = df[features]
y = df[target]


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

print("\nSplitting dataset chronologically...")

df["TIME"] = pd.to_datetime(df["TIME"])

split_date = df["TIME"].quantile(0.80)

train_mask = df["TIME"] < split_date
test_mask = df["TIME"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Split date:", split_date)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. DEFINE MODELS
# ==========================================

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


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

results = []
trained_models = {}

os.makedirs("models", exist_ok=True)

print("\nTraining models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Rainfall cannot be negative
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    # Save individual models
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        os.path.join("models", filename)
    )

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nMODEL COMPARISON")
print("=" * 50)

print(results_df.to_string(index=False))


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/best_rainfall_model.pkl"
)

print("\nBest Model:", best_model_name)

print("\nAll models trained and saved successfully!")
