import logging
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_auto_arima(train_series, seasonal_period=12):
    """
    Fits a SARIMA model via a small grid search on p/q/P/Q values using statsmodels.

    Parameters:
    -----------
    train_series : pd.Series
        The historical training target values.
    seasonal_period : int
        The seasonality window. Default is 12 (Monthly profile).
        Use 7 for daily data with weekly patterns.
    """
    logging.info(f"🧠 Training SARIMA baseline with seasonal cycle m={seasonal_period}...")

    best_aic = np.inf
    best_model = None
    best_order = None
    best_seasonal_order = None

    # Light search for a stable model without requiring pmdarima.
    for p in range(0, 3):
        for q in range(0, 3):
            for P in range(0, 2):
                for Q in range(0, 2):
                    try:
                        candidate = SARIMAX(
                            train_series,
                            order=(p, 1, q),
                            seasonal_order=(P, 1, Q, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fitted = candidate.fit(disp=False)
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, 1, q)
                            best_seasonal_order = (P, 1, Q, seasonal_period)
                    except Exception:
                        continue

    if best_model is None:
        raise RuntimeError("Unable to fit a SARIMA model on the provided series.")

    logging.info(f"✅ Search complete. Best Fit: SARIMA{best_order} x {best_seasonal_order} | AIC={best_aic:.2f}")
    return best_model