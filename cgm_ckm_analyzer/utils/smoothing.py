"""
Smoothing utilities for metabolic data visualization.

Provides rolling average and Savitzky-Golay smoothing for time series data.
"""

import numpy as np
import pandas as pd
from typing import Union


def rolling_smooth(
    series: pd.Series,
    window: int = 5,
    min_periods: int = 1,
    center: bool = True
) -> pd.Series:
    """Apply rolling average smoothing to a time series.
    
    Args:
        series: Input time series data.
        window: Window size for rolling average.
        min_periods: Minimum observations required in window.
        center: If True, center the window on each point.
    
    Returns:
        Smoothed series with same index.
    """
    return series.rolling(window=window, min_periods=min_periods, center=center).mean()


def savgol_smooth(
    series: pd.Series,
    window: int = 15,
    polyorder: int = 2
) -> pd.Series:
    """Apply Savitzky-Golay filter for smoothing.
    
    The Savitzky-Golay filter fits a polynomial to each window of data,
    which preserves features better than simple moving average.
    Particularly good for CGM data visualization.
    
    Args:
        series: Input time series data.
        window: Window length for the filter (must be odd and > polyorder).
        polyorder: Order of polynomial to fit.
    
    Returns:
        Smoothed series with same index.
    
    Raises:
        ValueError: If window is even or less than polyorder.
    """
    from scipy.signal import savgol_filter
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Ensure window > polyorder
    if window <= polyorder:
        window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    
    # Handle series shorter than window
    if len(series) < window:
        window = len(series) if len(series) % 2 == 1 else len(series) - 1
        if window <= polyorder:
            return series  # Can't smooth, return as-is
    
    # Drop NaN for filtering, then reindex
    valid_mask = series.notna()
    valid_series = series[valid_mask]
    
    if len(valid_series) < window:
        return series
    
    smoothed_values = savgol_filter(valid_series.values, window, polyorder)
    result = series.copy()
    result[valid_mask] = smoothed_values
    
    return result


def exponential_smooth(
    series: pd.Series,
    alpha: float = 0.3,
    adjust: bool = True
) -> pd.Series:
    """Apply exponential weighted moving average smoothing.
    
    Args:
        series: Input time series data.
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.
        adjust: If True, adjust weights for the beginning of the series.
    
    Returns:
        Smoothed series with same index.
    """
    return series.ewm(alpha=alpha, adjust=adjust).mean()


def smooth_by_time_of_day(
    df: pd.DataFrame,
    value_column: str,
    window: int = 111,
    polyorder: int = 2
) -> pd.DataFrame:
    """Smooth data grouped by time of day (minute of day).
    
    This is the approach used in cgm_plot.py for creating smooth
    daily overlay plots showing typical glucose patterns.
    
    Args:
        df: DataFrame with 'timestamp' column and value column.
        value_column: Name of the column to smooth.
        window: Savitzky-Golay window size.
        polyorder: Polynomial order.
    
    Returns:
        DataFrame with 'minute_of_day' and smoothed quantiles.
    """
    from scipy.signal import savgol_filter
    
    # Extract minute of day (0-1439)
    df = df.copy()
    df['minute_of_day'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    
    # Calculate quantiles for each minute
    grouped = df.groupby('minute_of_day')[value_column]
    quantiles = grouped.quantile([0.10, 0.34, 0.50, 0.68, 0.90]).unstack()
    
    # Ensure window is odd and valid
    if window % 2 == 0:
        window += 1
    if len(quantiles) < window:
        window = len(quantiles) if len(quantiles) % 2 == 1 else len(quantiles) - 1
    
    if window <= polyorder or len(quantiles) < 4:
        return quantiles
    
    # Apply Savitzky-Golay to each quantile
    smoothed = quantiles.copy()
    for col in smoothed.columns:
        if smoothed[col].notna().sum() >= window:
            smoothed[col] = savgol_filter(
                smoothed[col].interpolate().values,
                window,
                polyorder
            )
    
    return smoothed
