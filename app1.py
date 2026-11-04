
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from scipy.stats import skew, kurtosis, ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="India Rainfall Analytics",
    page_icon="🌧️",
    layout="wide"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #e6edf3;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #e6edf3;
}
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATA
# =====================================================

@st.cache_data
def load_rainfall():

    ds = xr.open_dataset(
        "data/RF25_ind2025_rfp25.nc"
    )

    return ds


@st.cache_data
def load_processed_data():

    df = pd.read_csv(
        "data/rainfall_processed.csv"
    )

    df["TIME"] = pd.to_datetime(df["TIME"])

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
def load_model_results():

    return pd.read_csv(
        "models/model_comparison.csv"
    )


ds = load_rainfall()
rain = ds["RAINFALL"]

df = load_processed_data()


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🌧️ Rainfall Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Spatial Analysis",
        "Statistical Analysis",
        "Probability Analysis",
        "Hypothesis Testing",
        "Machine Learning",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "IMD Gridded Daily Rainfall Data | India | 2025"
)

# Date selection available on pages that need it
selected_date = st.sidebar.date_input(
    "Select Date",
    value=pd.Timestamp("2025-08-15").date(),
    min_value=pd.Timestamp("2025-01-01").date(),
    max_value=pd.Timestamp("2025-12-31").date()
)


# =====================================================
# 5. GLOBAL VARIABLES
# =====================================================

valid_rain = df["Rainfall"].dropna()

rain_day = rain.sel(
    TIME=np.datetime64(selected_date)
)

rain_values = rain_day.values

valid_day = rain_values[
    ~np.isnan(rain_values)
]


# =====================================================
# 6. OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.title("🇮🇳 India Rainfall Analytics")

    st.markdown(
        "### Probability, Statistics & Machine Learning"
    )

    st.caption(
        "Exploring daily gridded rainfall patterns "
        "across India using IMD data."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Rainfall",
            f"{valid_rain.mean():.2f} mm"
        )

    with col2:
        st.metric(
            "Median Rainfall",
            f"{valid_rain.median():.2f} mm"
        )

    with col3:
        st.metric(
            "Maximum Rainfall",
            f"{valid_rain.max():.2f} mm"
        )

    with col4:
        st.metric(
            "Rainfall > 0 mm",
            f"{(valid_rain > 0).mean() * 100:.2f}%"
        )

    st.subheader(
        f"Daily Rainfall Distribution — "
        f"{selected_date.strftime('%d %B %Y')}"
    )

    fig = go.Figure(
        go.Heatmap(
            z=rain_day.values,
            x=rain_day["LONGITUDE"].values,
            y=rain_day["LATITUDE"].values,
            colorscale="Blues",
            colorbar=dict(title="Rainfall (mm)"),
            hovertemplate=(
                "Longitude: %{x}<br>"
                "Latitude: %{y}<br>"
                "Rainfall: %{z:.2f} mm"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_title="Longitude",
        yaxis_title="Latitude"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# 7. SPATIAL ANALYSIS PAGE
# =====================================================

elif page == "Spatial Analysis":

    st.title("🗺️ Spatial Rainfall Analysis")

    st.markdown(
        "Explore rainfall intensity across India's "
        "geographical grid."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Daily Average",
            f"{np.mean(valid_day):.2f} mm"
        )

    with col2:

        st.metric(
            "Daily Maximum",
            f"{np.max(valid_day):.2f} mm"
        )

    fig = go.Figure(
        go.Heatmap(
            z=rain_day.values,
            x=rain_day["LONGITUDE"].values,
            y=rain_day["LATITUDE"].values,
            colorscale="Turbo",
            colorbar=dict(title="Rainfall (mm)")
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        title=f"India Rainfall Map — {selected_date}",
        xaxis_title="Longitude",
        yaxis_title="Latitude"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Each cell represents a 0.25° × 0.25° "
        "gridded rainfall observation."
    )


# =====================================================
# 8. STATISTICAL ANALYSIS PAGE
# =====================================================


# =====================================================
# 8. STATISTICAL ANALYSIS PAGE
# =====================================================

elif page == "Statistical Analysis":

    st.title("📊 Statistical Analysis")
    st.markdown(
        "Explore rainfall distribution, variability, "
        "outliers, and temporal patterns."
    )

    # ---------------------------------------------
    # DESCRIPTIVE STATISTICS
    # ---------------------------------------------

    mean_val = valid_rain.mean()
    median_val = valid_rain.median()
    std_val = valid_rain.std()
    var_val = valid_rain.var()
    skew_val = skew(valid_rain)
    kurt_val = kurtosis(valid_rain)

    q25 = valid_rain.quantile(0.25)
    q75 = valid_rain.quantile(0.75)
    q95 = valid_rain.quantile(0.95)
    q99 = valid_rain.quantile(0.99)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Mean", f"{mean_val:.2f} mm")
    c2.metric("Median", f"{median_val:.2f} mm")
    c3.metric("Std Deviation", f"{std_val:.2f} mm")
    c4.metric("Skewness", f"{skew_val:.2f}")

    st.divider()

    # ---------------------------------------------
    # STATISTICS TABLE
    # ---------------------------------------------

    st.subheader("📋 Descriptive Summary")

    summary = pd.DataFrame({
        "Statistic": [
            "Count",
            "Mean",
            "Median",
            "Standard Deviation",
            "Variance",
            "Minimum",
            "25th Percentile",
            "75th Percentile",
            "95th Percentile",
            "99th Percentile",
            "Maximum",
            "Skewness",
            "Excess Kurtosis"
        ],
        "Value": [
            valid_rain.count(),
            mean_val,
            median_val,
            std_val,
            var_val,
            valid_rain.min(),
            q25,
            q75,
            q95,
            q99,
            valid_rain.max(),
            skew_val,
            kurt_val
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------
    # RAINFALL HISTOGRAM
    # ---------------------------------------------

    st.subheader("📈 Rainfall Frequency Distribution")

    hist_option = st.radio(
        "Distribution view",
        [
            "All rainfall values",
            "Positive rainfall only"
        ],
        horizontal=True
    )

    if hist_option == "All rainfall values":

        hist_data = valid_rain

    else:

        hist_data = valid_rain[valid_rain > 0]

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=80,
            marker_color="#38bdf8",
            name="Rainfall",
            opacity=0.85
        )
    )

    fig_hist.update_layout(
        template="plotly_dark",
        title="Rainfall Frequency Distribution",
        xaxis_title="Rainfall (mm)",
        yaxis_title="Frequency",
        height=500,
        bargap=0.05
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.caption(
        "Rainfall is strongly right-skewed. "
        "A large number of zero-rain observations "
        "and a small number of extreme events "
        "produce this distribution."
    )

    # ---------------------------------------------
    # BOX PLOT WITH OUTLIER CONTROL
    # ---------------------------------------------

    st.subheader("📦 Rainfall Box Plot")

    box_limit = st.selectbox(
        "Display range",
        [
            "Full range",
            "Up to 95th percentile",
            "Up to 99th percentile"
        ]
    )

    if box_limit == "Full range":

        box_data = valid_rain

    elif box_limit == "Up to 95th percentile":

        box_data = valid_rain[
            valid_rain <= q95
        ]

    else:

        box_data = valid_rain[
            valid_rain <= q99
        ]

    fig_box = go.Figure()

    fig_box.add_trace(
        go.Box(
            y=box_data,
            name="Rainfall",
            boxpoints=False,
            marker_color="#a78bfa",
            fillcolor="rgba(167,139,250,0.35)"
        )
    )

    fig_box.update_layout(
        template="plotly_dark",
        title="Rainfall Spread and Quartiles",
        yaxis_title="Rainfall (mm)",
        height=500
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

    st.caption(
        f"Showing {len(box_data):,} observations. "
        "IQR = Q3 − Q1 = {q75 - q25:.2f} mm."
    )

    # ---------------------------------------------
    # MONTHLY RAINFALL BOXPLOT
    # ---------------------------------------------

    st.subheader("🗓️ Monthly Rainfall Variability")

    monthly_df = df.copy()

    monthly_df["Month_Name"] = monthly_df[
        "TIME"
    ].dt.strftime("%b")

    month_order = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    fig_monthly_box = px.box(
        monthly_df,
        x="Month_Name",
        y="Rainfall",
        category_orders={
            "Month_Name": month_order
        },
        color="Month_Name",
        title="Monthly Distribution of Daily Rainfall",
        points=False
    )

    fig_monthly_box.update_layout(
        template="plotly_dark",
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        height=550,
        showlegend=False
    )

    st.plotly_chart(
        fig_monthly_box,
        use_container_width=True
    )

    # ---------------------------------------------
    # EMPIRICAL CDF
    # ---------------------------------------------

    st.subheader("📉 Empirical Cumulative Distribution")

    # Sort values
    sorted_rain = np.sort(
        valid_rain.values
    )

    # Downsample for rendering performance
    step = max(
        1,
        len(sorted_rain) // 10000
    )

    cdf_x = sorted_rain[::step]

    cdf_y = (
        np.arange(1, len(sorted_rain) + 1)
        / len(sorted_rain)
    )[::step]

    fig_cdf = go.Figure()

    fig_cdf.add_trace(
        go.Scatter(
            x=cdf_x,
            y=cdf_y,
            mode="lines",
            line=dict(
                color="#34d399",
                width=2
            ),
            name="Empirical CDF"
        )
    )

    fig_cdf.update_layout(
        template="plotly_dark",
        title="Empirical Cumulative Distribution Function (CDF)",
        xaxis_title="Rainfall (mm)",
        yaxis_title="Cumulative Probability",
        yaxis=dict(
            tickformat=".0%"
        ),
        height=500
    )

    st.plotly_chart(
        fig_cdf,
        use_container_width=True
    )

    # ---------------------------------------------
    # MONTHLY MEAN AND STANDARD DEVIATION
    # ---------------------------------------------

    st.subheader("📊 Monthly Mean Rainfall with Variability")

    monthly_stats = (
        df.groupby("Month")["Rainfall"]
        .agg(["mean", "std"])
        .reset_index()
    )

    monthly_stats["Month_Name"] = monthly_stats[
        "Month"
    ].apply(
        lambda x: pd.Timestamp(
            month=x,
            day=1,
            year=2025
        ).strftime("%B")
    )

    fig_monthly = go.Figure()

    fig_monthly.add_trace(
        go.Bar(
            x=monthly_stats["Month_Name"],
            y=monthly_stats["mean"],
            name="Mean Rainfall",
            marker_color="#38bdf8"
        )
    )

    fig_monthly.add_trace(
        go.Scatter(
            x=monthly_stats["Month_Name"],
            y=monthly_stats["mean"],
            error_y=dict(
                type="data",
                array=monthly_stats["std"],
                visible=True
            ),
            mode="markers",
            name="± 1 Standard Deviation",
            marker_color="#fbbf24"
        )
    )

    fig_monthly.update_layout(
        template="plotly_dark",
        title="Monthly Mean Rainfall ± Standard Deviation",
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        height=500
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )


# =====================================================
# 9. PROBABILITY ANALYSIS PAGE
# =====================================================

elif page == "Probability Analysis":

    st.title("🎲 Probability Analysis")

    st.markdown(
        "Empirical probabilities calculated from "
        "observed daily rainfall."
    )

    thresholds = [0, 1, 5, 10, 25, 50, 100]

    probabilities = []

    for threshold in thresholds:

        probability = np.mean(
            valid_rain > threshold
        )

        probabilities.append({
            "Threshold (mm)": threshold,
            "Probability": probability,
            "Percentage": probability * 100
        })

    prob_df = pd.DataFrame(probabilities)

    st.subheader("Rainfall Exceedance Probability")

    fig_prob = px.bar(
        prob_df,
        x="Threshold (mm)",
        y="Percentage",
        color="Percentage",
        color_continuous_scale="Blues",
        title="Probability of Rainfall Exceeding Threshold"
    )

    fig_prob.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig_prob,
        use_container_width=True
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )

    # Empirical CDF
    st.subheader("Empirical Cumulative Distribution Function")

    sorted_rain = np.sort(valid_rain.values)

    cumulative_probability = (
        np.arange(1, len(sorted_rain) + 1)
        / len(sorted_rain)
    )

    fig_cdf = go.Figure()

    fig_cdf.add_trace(
        go.Scatter(
            x=sorted_rain,
            y=cumulative_probability,
            mode="lines",
            name="Empirical CDF",
            line=dict(color="#00b4d8", width=2)
        )
    )

    fig_cdf.update_layout(
        template="plotly_dark",
        title="Empirical CDF of Daily Rainfall",
        xaxis_title="Rainfall (mm)",
        yaxis_title="Cumulative Probability",
        height=500
    )

    st.plotly_chart(
        fig_cdf,
        use_container_width=True
    )


# =====================================================
# 10. HYPOTHESIS TESTING PAGE
# =====================================================

elif page == "Hypothesis Testing":

    st.title("🧪 Hypothesis Testing")

    st.markdown(
        "Compare rainfall between two selected months "
        "using a Welch independent-samples t-test."
    )

    month1 = st.selectbox(
        "Select First Month",
        list(range(1, 13)),
        index=5,
        format_func=lambda x: pd.Timestamp(
            month=x, day=1, year=2025
        ).strftime("%B")
    )

    month2 = st.selectbox(
        "Select Second Month",
        list(range(1, 13)),
        index=8,
        format_func=lambda x: pd.Timestamp(
            month=x, day=1, year=2025
        ).strftime("%B")
    )

    sample1 = df.loc[
        df["Month"] == month1,
        "Rainfall"
    ]

    sample2 = df.loc[
        df["Month"] == month2,
        "Rainfall"
    ]

    if month1 != month2:

        t_stat, p_value = __import__(
            "scipy.stats",
            fromlist=["ttest_ind"]
        ).ttest_ind(
            sample1,
            sample2,
            equal_var=False
        )

        mean1 = sample1.mean()
        mean2 = sample2.mean()

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mean — Month 1",
            f"{mean1:.3f} mm"
        )

        col2.metric(
            "Mean — Month 2",
            f"{mean2:.3f} mm"
        )

        col3.metric(
            "Mean Difference",
            f"{mean1 - mean2:.3f} mm"
        )

        st.subheader("Test Results")

        st.write(f"t-statistic: **{t_stat:.4f}**")
        st.write(f"p-value: **{p_value:.8f}**")

        if p_value < 0.05:

            st.success(
                "Reject H₀: The monthly mean rainfall "
                "difference is statistically significant."
            )

        else:

            st.warning(
                "Fail to reject H₀: Insufficient evidence "
                "of a difference in monthly mean rainfall."
            )

        fig_test = px.box(
            pd.DataFrame({
                "Rainfall": pd.concat(
                    [sample1, sample2]
                ),
                "Month": (
                    [pd.Timestamp(month=month1, day=1, year=2025).strftime("%B")]
                    * len(sample1)
                    +
                    [pd.Timestamp(month=month2, day=1, year=2025).strftime("%B")]
                    * len(sample2)
                )
            }),
            x="Month",
            y="Rainfall",
            color="Month",
            title="Rainfall Comparison Between Months"
        )

        fig_test.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig_test,
            use_container_width=True
        )

    else:

        st.warning(
            "Please select two different months."
        )


# =====================================================
# 11. MACHINE LEARNING PAGE
# =====================================================

elif page == "Machine Learning":

    st.title("🤖 Rainfall Prediction")

    st.markdown(
        "Predict rainfall using the trained regression models."
    )

    models = load_models()

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=6.5,
            max_value=38.5,
            value=15.0
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            min_value=66.5,
            max_value=100.0,
            value=75.0
        )

    prediction_date = st.date_input(
        "Prediction Date",
        value=pd.Timestamp("2025-08-15").date(),
        min_value=pd.Timestamp("2025-01-01").date(),
        max_value=pd.Timestamp("2025-12-31").date()
    )

    date_obj = pd.Timestamp(prediction_date)

    month = date_obj.month
    day_of_year = date_obj.dayofyear

    input_data = pd.DataFrame({
        "LATITUDE": [latitude],
        "LONGITUDE": [longitude],
        "Month": [month],
        "DayOfYear": [day_of_year],
        "Day_sin": [
            np.sin(2 * np.pi * day_of_year / 365)
        ],
        "Day_cos": [
            np.cos(2 * np.pi * day_of_year / 365)
        ]
    })

    if st.button("Predict Rainfall", type="primary"):

        prediction_results = []

        for name, model in models.items():

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(prediction, 0)

            prediction_results.append({
                "Model": name,
                "Predicted Rainfall (mm)": prediction
            })

        prediction_df = pd.DataFrame(
            prediction_results
        )

        st.subheader("Prediction Results")

        st.dataframe(
            prediction_df,
            use_container_width=True
        )

        fig_pred = px.bar(
            prediction_df,
            x="Model",
            y="Predicted Rainfall (mm)",
            color="Model",
            title="Model Prediction Comparison"
        )

        fig_pred.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig_pred,
            use_container_width=True
        )

        st.warning(
            "Experimental predictions only. Models were trained "
            "on 2025 data and do not include atmospheric "
            "weather variables."
        )


# =====================================================
# 12. MODEL EVALUATION PAGE
# =====================================================

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    results = load_model_results()

    st.subheader("Performance Comparison")

    fig_metrics = px.bar(
        results,
        x="Model",
        y="RMSE",
        color="Model",
        title="RMSE Comparison — Lower Is Better"
    )

    fig_metrics.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig_metrics,
        use_container_width=True
    )

    fig_mae = px.bar(
        results,
        x="Model",
        y="MAE",
        color="Model",
        title="MAE Comparison — Lower Is Better"
    )

    fig_mae.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

    fig_r2 = px.bar(
        results,
        x="Model",
        y="R2 Score",
        color="Model",
        title="R² Score Comparison"
    )

    fig_r2.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )

    st.subheader("Complete Metrics Table")

    st.dataframe(
        results.sort_values("RMSE"),
        use_container_width=True
    )

    st.info(
        "Linear Regression performed best on this test set. "
        "Low R² scores indicate limited predictive skill."
    )


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "India Rainfall Analytics | Probability & Statistics "
    "Academic Project | IMD Data | 2025"
)
