
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rainfall Analytics | India",
    page_icon="🌧️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f5f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #123456;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5eaf0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():

    data_path = "data/rainfall_districtwise_daily_imd.csv"

    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df.rename(
        columns={"Daily Actual": "Rainfall"},
        inplace=True
    )

    df.dropna(
        subset=["Date", "Rainfall"],
        inplace=True
    )

    return df


@st.cache_resource
def load_models():

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

    return models


@st.cache_data
def load_comparison():

    return pd.read_csv(
        "models/model_comparison.csv"
    )


df = load_data()

models = load_models()

comparison = load_comparison()


# ============================================================
# HEADER
# ============================================================

st.title("🌧️ India Rainfall Analytics")

st.markdown(
    "### Machine Learning Powered Rainfall Prediction"
)

st.markdown(
    """
    Explore rainfall patterns across Indian states and districts,
    compare machine learning models, and generate rainfall predictions.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Prediction Settings")

states = sorted(
    df["State"].dropna().unique()
)

selected_state = st.sidebar.selectbox(
    "Select State",
    states
)


districts = sorted(
    df[df["State"] == selected_state]["District"]
    .dropna()
    .unique()
)

selected_district = st.sidebar.selectbox(
    "Select District",
    districts
)


selected_date = st.sidebar.date_input(
    "Select Prediction Date"
)


st.sidebar.info(
    "Predictions are based on historical rainfall "
    "patterns and seasonal features."
)


# ============================================================
# FEATURE ENGINEERING FOR PREDICTION
# ============================================================

month = selected_date.month

day_of_year = selected_date.timetuple().tm_yday

month_sin = np.sin(
    2 * np.pi * month / 12
)

month_cos = np.cos(
    2 * np.pi * month / 12
)

day_sin = np.sin(
    2 * np.pi * day_of_year / 365.25
)

day_cos = np.cos(
    2 * np.pi * day_of_year / 365.25
)


input_data = pd.DataFrame({

    "State": [selected_state],

    "District": [selected_district],

    "Month_sin": [month_sin],

    "Month_cos": [month_cos],

    "DayOfYear_sin": [day_sin],

    "DayOfYear_cos": [day_cos]

})


# ============================================================
# PREDICTIONS FROM ALL MODELS
# ============================================================

predictions = {}

for name, model in models.items():

    prediction = model.predict(input_data)[0]

    # Rainfall cannot be negative
    prediction = max(0, prediction)

    predictions[name] = prediction


best_model_name = comparison.iloc[0]["Model"]

best_prediction = predictions[
    best_model_name
]


# ============================================================
# MAIN PREDICTION DISPLAY
# ============================================================

st.subheader("📍 Rainfall Prediction")

st.write(
    f"Location: **{selected_district}, {selected_state}**"
)

st.write(
    f"Prediction Date: **{selected_date.strftime('%d %B %Y')}**"
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Linear Regression",
    f"{predictions['Linear Regression']:.2f} mm"
)

col2.metric(
    "Random Forest",
    f"{predictions['Random Forest']:.2f} mm"
)

col3.metric(
    "XGBoost",
    f"{predictions['XGBoost']:.2f} mm"
)


st.success(
    f"Best Model: {best_model_name} | "
    f"Predicted Rainfall: {best_prediction:.2f} mm"
)


st.divider()


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.subheader("📊 Model Performance Comparison")

metric1, metric2, metric3 = st.columns(3)

best_mae = comparison["MAE"].min()

best_rmse = comparison["RMSE"].min()

best_r2 = comparison["R2 Score"].max()


metric1.metric(
    "Best MAE",
    f"{best_mae:.3f} mm"
)

metric2.metric(
    "Best RMSE",
    f"{best_rmse:.3f} mm"
)

metric3.metric(
    "Best R² Score",
    f"{best_r2:.4f}"
)


# ============================================================
# PERFORMANCE CHART
# ============================================================

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=comparison["Model"],
        y=comparison["MAE"],
        name="MAE",
        marker_color="#3498db"
    )
)

fig.add_trace(
    go.Bar(
        x=comparison["Model"],
        y=comparison["RMSE"],
        name="RMSE",
        marker_color="#e67e22"
    )
)

fig.update_layout(
    title="Model Error Comparison",
    xaxis_title="Machine Learning Model",
    yaxis_title="Error (mm)",
    barmode="group",
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# R2 SCORE COMPARISON
# ============================================================

fig_r2 = px.bar(
    comparison,
    x="Model",
    y="R2 Score",
    color="Model",
    title="R² Score Comparison",
    template="plotly_white"
)

fig_r2.update_layout(
    height=400,
    showlegend=False
)

st.plotly_chart(
    fig_r2,
    use_container_width=True
)


# ============================================================
# MODEL RESULTS TABLE
# ============================================================

st.subheader("📋 Detailed Model Results")

st.dataframe(
    comparison.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HISTORICAL RAINFALL ANALYSIS
# ============================================================

st.divider()

st.subheader("📈 Historical Rainfall Analysis")


district_data = df[
    (df["State"] == selected_state) &
    (df["District"] == selected_district)
].copy()


district_data = district_data.sort_values(
    "Date"
)


if not district_data.empty:

    historical_fig = px.line(
        district_data,
        x="Date",
        y="Rainfall",
        title=f"Historical Daily Rainfall — {selected_district}",
        labels={
            "Rainfall": "Rainfall (mm)",
            "Date": "Date"
        },
        template="plotly_white"
    )

    historical_fig.update_layout(
        height=450
    )

    st.plotly_chart(
        historical_fig,
        use_container_width=True
    )


# ============================================================
# MONTHLY RAINFALL TREND
# ============================================================

df["Month"] = df["Date"].dt.month

monthly_data = (
    df.groupby("Month")["Rainfall"]
    .mean()
    .reset_index()
)

monthly_data["Month Name"] = monthly_data["Month"].apply(
    lambda x: pd.Timestamp(
        month=x,
        day=1,
        year=2024
    ).strftime("%B")
)


monthly_fig = px.line(
    monthly_data,
    x="Month Name",
    y="Rainfall",
    markers=True,
    title="Average Monthly Rainfall Across India",
    labels={
        "Rainfall": "Average Rainfall (mm)",
        "Month Name": "Month"
    },
    template="plotly_white"
)

monthly_fig.update_layout(
    height=400
)

st.plotly_chart(
    monthly_fig,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "India Rainfall Analytics | "
    "Machine Learning + Probability & Statistics Project"
)
