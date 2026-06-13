import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Import our custom modules
from src.data_loader import load_and_clean_csv, split_data
from src.eda import plot_time_series, decompose_series
from src.models import fit_auto_arima

# Configure logging for production tracing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_forecast(actual, predicted):
    """Calculates evaluation metrics for the forecast."""
    mape = mean_absolute_percentage_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mape, rmse

def plot_and_save(train, test, forecast, filepath="outputs/forecast_plot.png"):
    """Generates and saves a publication-quality forecast visualization."""
    plt.figure(figsize=(12, 6))
    
    # Plot historical, actual test, and predicted lines
    plt.plot(train.index, train["Sales"], label="Historical Train Data", color="#1f77b4", linewidth=2)
    plt.plot(test.index, test["Sales"], label="Actual Sales (Test Period)", color="#2ca02c", linewidth=2)
    plt.plot(forecast.index, forecast, label="Auto-ARIMA Forecast", color="#d62728", linestyle="--", linewidth=2)
    
    # Clean up charts aesthetics
    plt.title("Sales Forecasting Project: Production Run", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Sales Volume / Units", fontsize=12)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    # Save chart safely
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"💾 Final forecast visualization saved to '{filepath}'")

def generate_fallback_data():
    """Generates synthetic data if a real CSV is missing to keep the pipeline runnable."""
    logging.warning("⚠️ Real CSV data target not found. Initializing mock engine for project demonstration...")
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2025-12-01", freq="MS")
    trend = np.linspace(100, 250, len(dates))
    seasonality = np.array([10, 5, 15, 20, 25, 30, 15, 10, 20, 35, 50, 90] * 5)
    noise = np.random.normal(0, 10, len(dates))
    
    df = pd.DataFrame({"Sales": trend + seasonality + noise}, index=dates)
    df.index.name = "Date"
    return df

def main():
    logging.info("🚀 Starting Advanced Sales Forecasting Pipeline...")
    
    # Target path configurations
    CSV_PATH = "data/raw/sales_data.csv" 
    DATE_HEADER = "Order_Date"
    SALES_HEADER = "Revenue"
    
    # 1. Load Data (with automated fallback handling)
    if os.path.exists(CSV_PATH):
        df = load_and_clean_csv(filepath=CSV_PATH, date_col=DATE_HEADER, sales_col=SALES_HEADER, freq="MS")
    else:
        df = generate_fallback_data()
        
    logging.info(f"📋 Dataset active. Shape: {df.shape} | Time Span: {df.index.min().date()} to {df.index.max().date()}")
    
    # 2. Run Exploratory Data Analysis (EDA)
    plot_time_series(df)
    decompose_series(df)
    
    # 3. Train-Test Split
    TEST_WINDOW = 12
    train, test = split_data(df, test_months=TEST_WINDOW)
    logging.info(f"✂️ Data split complete. Training steps: {len(train)}, Testing steps: {len(test)}")
    
    # 4. Automated Model Tuning & Training
    try:
        auto_model = fit_auto_arima(train["Sales"], seasonal_period=12)
    except Exception as e:
        logging.error(f"❌ Pipeline halted due to training optimization failure: {str(e)}")
        return

    # 5. Generate Future Forecast
    logging.info(f"🔮 Projecting sales out {TEST_WINDOW} months into the future...")
    forecast_values = auto_model.predict(n_periods=TEST_WINDOW)
    forecast_series = pd.Series(forecast_values, index=test.index)
    
    # 6. Pipeline Evaluation Metrics
    mape, rmse = evaluate_forecast(test["Sales"], forecast_series)
    
    print("\n" + "="*40)
    print("📈 PIPELINE METRICS EVALUATION")
    print("="*40)
    print(f"  • MAPE (Mean Absolute % Error): {mape:.2%}")
    print(f"  • RMSE (Root Mean Squared Error): {rmse:.2f}")
    print("="*40 + "\n")
    
    # 7. Export Visual Deliverables
    plot_and_save(train, test, forecast_series)
    logging.info("🏁 Pipeline executed cleanly. Visual graphs are available inside the 'outputs/' folder.")

if __name__ == "__main__":
    main()