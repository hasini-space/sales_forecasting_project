import pandas as pd
import numpy as np
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not installed. Use 'pip install yfinance' to enable Yahoo Finance data fetching.")

def fetch_yfinance_data(ticker, start_date, end_date, interval="1d", price_column="Close"):
    """
    Fetches historical financial data from Yahoo Finance using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', '^GSPC' for S&P 500).
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    interval : str
        Data frequency. Default is '1d' (daily). Options: '1m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk', '1mo'.
    price_column : str
        Column to use as the target series. Default is 'Close'. Options: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'Sales' column containing the selected price data, indexed by date.
    
    Example:
    --------
    >>> df = fetch_yfinance_data('AAPL', '2023-01-01', '2024-12-31', interval='1d', price_column='Close')
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Run 'pip install yfinance' first.")
    
    logging.info(f"📡 Fetching {ticker} data from Yahoo Finance ({start_date} to {end_date}, interval={interval})...")
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            raise ValueError(f"No data returned for ticker '{ticker}' in the specified date range.")
        
        # Handle single vs. multiple ticker download (yfinance returns different structures)
        if isinstance(data.columns, pd.MultiIndex):
            # Multiple tickers: extract the specific ticker's data
            data = data[ticker]
        
        # Extract the target price column and rename to 'Sales'
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found. Available columns: {list(data.columns)}")
        
        result_df = pd.DataFrame(index=data.index)
        result_df["Sales"] = data[price_column]
        result_df.index.name = "Date"
        
        logging.info(f"✅ Successfully fetched {len(result_df)} data points for {ticker}.")
        return result_df
    
    except Exception as e:
        logging.error(f"❌ Error fetching data from Yahoo Finance: {str(e)}")
        raise

def load_and_clean_csv(filepath, date_col, sales_col, freq="MS"):

    """
    Loads a raw sales CSV, handles formatting anomalies, treats missing 
    values, and aggregates data to a uniform time-series frequency.
    
    Parameters:
    -----------
    filepath : str
        Path to the target CSV file (e.g., 'data/raw/sales_data.csv').
    date_col : str
        The name of the timestamp/date column in the CSV.
    sales_col : str
        The name of the target sales/revenue volume column.
    freq : str
        Target resampling frequency. Default is 'MS' (Month Start). 
        Use 'W' for weekly or 'D' for daily tracking.
    """
    logging.info(f"📂 Reading raw data from: {filepath}")
    
    try:
        # 1. Load data, forcing specific target columns to strings for uniform processing
        df = pd.read_csv(filepath, usecols=[date_col, sales_col])
        
        # 2. Convert and standardize Date Column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows where dates couldn't be parsed
        if df[date_col].isna().sum() > 0:
            logging.warning(f"⚠️ Dropped {df[date_col].isna().sum()} rows due to unparseable dates.")
            df = df.dropna(subset=[date_col])
            
        # 3. Clean and cast Sales Column to float
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
        df = df.dropna(subset=[sales_col])
        
        # 4. Set chronological index
        df = df.set_index(date_col).sort_index()
        
        # 5. Aggregate to target frequency (sums up transactions if data is granular)
        logging.info(f"🔄 Resampling time series data to frequency: '{freq}'")
        resampled_series = df[sales_col].resample(freq).sum()
        
        # 6. Fill structural gaps (missing weeks/months in sequence)
        # Reindexing introduces NaN for missing intervals, which we fix with linear interpolation
        full_idx = pd.date_range(start=resampled_series.index.min(), end=resampled_series.index.max(), freq=freq)
        cleaned_df = pd.DataFrame(index=full_idx)
        cleaned_df["Sales"] = resampled_series.reindex(full_idx)
        
        missing_intervals = cleaned_df["Sales"].isna().sum()
        if missing_intervals > 0:
            logging.info(f"🔧 Imputing {missing_intervals} missing time steps using linear interpolation.")
            cleaned_df["Sales"] = cleaned_df["Sales"].interpolate(method='linear')
            
        logging.info(f"✅ Preprocessing successful. Cleaned dataset contains {len(cleaned_df)} intervals.")
        return cleaned_df

    except FileNotFoundError:
        logging.error(f"❌ Error: File not found at '{filepath}'. Verify directory layout.")
        raise
    except Exception as e:
        logging.error(f"❌ Critical pipeline failure during ingestion: {str(e)}")
        raise

def split_data(df, test_months=12):
    """Splits time series data into historical training and future verification windows."""
    if len(df) <= test_months:
        raise ValueError(f"Dataset length ({len(df)}) must be larger than the test window ({test_months}).")
        
    train = df.iloc[:-test_months]
    test = df.iloc[-test_months:]
    return train, test