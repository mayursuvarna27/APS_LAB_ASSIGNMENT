
import pandas as pd
import numpy as np
import joblib

from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading processed dataset...")

df = pd.read_csv("data/rainfall_processed.csv")
df["TIME"] = pd.to_datetime(df["TIME"])

features = [
    "LATITUDE",
    "LONGITUDE",
    "Month",
    "DayOfYear",
    "Day_sin",
    "Day_cos"
]

X = df[features]
y = df["Rainfall"]

# Same chronological split as train2.py
split_date = df["TIME"].quantile(0.80)

train_mask = df["TIME"] < split_date
test_mask = df["TIME"] >= split_date

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("Test samples:", len(X_test))


# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

models = {
    "Linear Regression": joblib.load(
        "models/linear_regression.pkl"
    ),

    "Random Forest": joblib.load(
        "models/random_forest.pkl"
    ),

    "XGBoost": joblib.load(
        "models/xgboost.pkl"
    )
}


# ==========================================
# 3. GENERATE PREDICTIONS
# ==========================================

predictions = {}

for name, model in models.items():

    pred = model.predict(X_test)

    # Rainfall cannot be negative
    pred = np.maximum(pred, 0)

    predictions[name] = pred


# ==========================================
# 4. CALCULATE ABSOLUTE ERRORS
# ==========================================

errors = {}

for name, pred in predictions.items():

    errors[name] = np.abs(y_test.values - pred)


# ==========================================
# 5. PAIRED T-TEST
# ==========================================

print("\nPAIRED T-TEST")
print("=" * 60)

model_pairs = [
    ("Linear Regression", "Random Forest"),
    ("Linear Regression", "XGBoost"),
    ("Random Forest", "XGBoost")
]

alpha = 0.05

for model_a, model_b in model_pairs:

    error_a = errors[model_a]
    error_b = errors[model_b]

    # Difference in absolute errors
    differences = error_a - error_b

    t_stat, p_value = ttest_rel(
        error_a,
        error_b
    )

    mean_diff = np.mean(differences)

    std_diff = np.std(differences, ddof=1)

    n = len(differences)

    # 95% confidence interval
    from scipy.stats import t

    critical_value = t.ppf(
        1 - alpha / 2,
        df=n - 1
    )

    margin_error = critical_value * (
        std_diff / np.sqrt(n)
    )

    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error

    # Cohen's dz effect size
    cohens_dz = mean_diff / std_diff

    print(f"\n{model_a} vs {model_b}")

    print(f"Mean error difference: {mean_diff:.6f}")

    print(
        f"95% CI: "
        f"[{ci_lower:.6f}, {ci_upper:.6f}]"
    )

    print(f"t-statistic: {t_stat:.6f}")
    print(f"p-value: {p_value:.10f}")

    print(f"Cohen's dz: {cohens_dz:.6f}")

    if p_value < alpha:
        print("Conclusion: Statistically significant difference")
    else:
        print("Conclusion: No statistically significant difference")


# ==========================================
# 6. WILCOXON SIGNED-RANK TEST
# ==========================================

print("\nWILCOXON SIGNED-RANK TEST")
print("=" * 60)

for model_a, model_b in model_pairs:

    error_a = errors[model_a]
    error_b = errors[model_b]

    try:

        w_stat, p_value = wilcoxon(
            error_a,
            error_b,
            alternative="two-sided"
        )

        print(f"\n{model_a} vs {model_b}")

        print(f"W-statistic: {w_stat:.6f}")
        print(f"p-value: {p_value:.10f}")

        if p_value < alpha:
            print("Conclusion: Statistically significant difference")
        else:
            print("Conclusion: No statistically significant difference")

    except ValueError as e:

        print("Wilcoxon test could not be computed:", e)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\nOVERALL MAE")
print("=" * 40)

for name, error in errors.items():

    print(
        f"{name}: "
        f"{np.mean(error):.6f}"
    )

print("\nStatistical evaluation completed!")
