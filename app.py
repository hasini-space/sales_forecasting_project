import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Import our custom pipeline modules
from src.data_loader import load_and_clean_csv, split_data
from src.models import fit_auto_arima

# Page Configuration for a widescreen, modern layout
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling injection for clean metrics and polished layout
st.markdown("""
    <style>
    .metric-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-title { font-size: 14px; color: #6c757d; font-weight: bold; }
    .metric-value { font-size: 28px; color: #1f77b4; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🛠️ Dashboard Configurations")

# 1. File Uploader Layout
uploaded_file = st.sidebar.file_uploader("Upload Raw Sales CSV", type=["csv"])

# Default fallback configurations mapped to our standard documentation variables
date_col = st.sidebar.text_input("Date Column Header", value="Order_Date")
sales_col = st.sidebar.text_input("Sales Column Header", value="Revenue")
freq_setting = st.sidebar.selectbox("Data Aggregation Frequency", options=["MS", "W", "D"], format_func=lambda x: {"MS": "Monthly Start", "W": "Weekly", "D": "Daily"}[x])

# 2. Interactive Horizon Parameters
test_window = st.sidebar.slider("Evaluation Test Window (Steps)", min_value=3, max_value=24, value=12)

# --- MAIN DASHBOARD INTERFACE ---
st.title("📈 Auto-ARIMA Sales Forecasting Engine")
st.markdown("An enterprise analytical application designed to ingest historic transactional datasets, isolate seasonal profiles, and project future performance thresholds.")
st.markdown("---")

if uploaded_file is not None:
    # Save uploaded payload securely into local directory structure
    raw_path = os.path.join("data", "raw", "uploaded_sales.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        # Pipeline Data Processing Stage
        with st.spinner("Processing structural timelines..."):
            df = load_and_clean_csv(filepath=raw_path, date_col=date_col, sales_col=sales_col, freq=freq_setting)
            train, test = split_data(df, test_months=test_window)
            
        st.success(f"Dataset successfully compiled! Extracted {len(df)} total intervals across history.")
        
        # Pipeline Algorithmic Training Stage
        with st.spinner("Optimizing SARIMA parameters via automated AIC grid search..."):
            m_val = 12 if freq_setting == "MS" else (7 if freq_setting == "D" else 52)
            model = fit_auto_arima(train["Sales"], seasonal_period=m_val)

            # Predict downstream metrics
            forecast_values = model.forecast(steps=test_window)
            forecast_series = pd.Series(forecast_values, index=test.index)

            # Calculate Validation KPI Analytics
            mape = mean_absolute_percentage_error(test["Sales"], forecast_series)
            rmse = np.sqrt(mean_squared_error(test["Sales"], forecast_series))

        # --- UI LAYOUT KPI CARDS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-title">MODEL CONFIG IDENTIFIED</div><div class="metric-value" style="font-size:20px;">SARIMA{model.order}x{model.seasonal_order}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box" style="border-left-color: #2ca02c;"><div class="metric-title">MAPE (ERROR %)</div><div class="metric-value">{mape:.2%}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box" style="border-left-color: #d62728;"><div class="metric-title">RMSE (ERROR MAGNITUDE)</div><div class="metric-value">{rmse:.2f}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- UI LAYOUT VISUALIZATION INTERFACE ---
        st.subheader("🔮 Predictive Forecasting Model Inversion")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(train.index, train["Sales"], label="Historical Baseline (Train)", color="#1f77b4", linewidth=2)
        ax.plot(test.index, test["Sales"], label="Actual Metrics (Test Window)", color="#2ca02c", linewidth=2)
        ax.plot(forecast_series.index, forecast_series, label="Algorithmic Extrapolation", color="#d62728", linestyle="--", linewidth=2)
        
        ax.set_xlabel("Chronological Axis")
        ax.set_ylabel("Sales Units / Value")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Raw Data Inspection Dropdowns
        with st.expander("🔎 View Extrapolated Matrix Predictions"):
            output_table = pd.DataFrame({
                "Actual Performance": test["Sales"],
                "Model Forecast": forecast_series,
                "Absolute Variance": np.abs(test["Sales"] - forecast_series)
            })
            st.dataframe(output_table.style.format("{:.2f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Execution Error within Data Matrix formatting pipeline: {e}")
        st.info("Ensure the column headers entered in the sidebar match your CSV file precisely.")

else:
    # Initial landing screen context warning
    st.info("👋 Welcome! Please upload a structured Sales .CSV file inside the left-hand sidebar control panel to kick off your real-time analytics engine.")