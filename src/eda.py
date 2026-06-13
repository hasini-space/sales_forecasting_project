import os
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure logging locally if not inherited
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def plot_time_series(df, column="Sales", filepath="outputs/eda_time_plot.png"):
    """Plots the raw time series to check for visible trends, outliers, and structural breaks."""
    logging.info("📊 Generating core time series EDA plot...")
    
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[column], color="#1f77b4", linewidth=2, label="Observed Sales")
    
    plt.title("Sales Volume Over Time (EDA)", fontsize=14, fontweight="bold")
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Units / Revenue", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"💾 EDA Time Plot saved to '{filepath}'")

def decompose_series(df, column="Sales", period=12, filepath="outputs/eda_decomposition.png"):
    """Decomposes the series into Trend, Seasonality, and Residual components."""
    logging.info(f"🔄 Performing seasonal decomposition (period={period})...")
    
    # Clean structural decomposition
    decomposition = seasonal_decompose(df[column], model='additive', period=period)
    
    # Plot components manually for clean control over sizing and labels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(decomposition.trend, color="#ff7f0e", linewidth=2)
    ax1.set_title("Long-Term Trend", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    ax2.plot(decomposition.seasonal, color="#2ca02c", linewidth=2)
    ax2.set_title("Seasonal Fluctuations", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    ax3.plot(decomposition.resid, color="#7f7f7f", linestyle="none", marker="o", markersize=4)
    ax3.set_title("Residuals (Noise)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    plt.suptitle("Time Series Structural Decomposition", fontsize=14, fontweight="bold")
    plt.xlabel("Timeline", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"💾 Decomposition plot saved to '{filepath}'")