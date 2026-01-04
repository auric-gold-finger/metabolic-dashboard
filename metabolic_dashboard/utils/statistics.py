"""
Statistical utilities for metabolic data analysis.

Provides common statistical calculations used across analyzers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional


def calculate_cv(values: Union[np.ndarray, pd.Series]) -> float:
    """Calculate coefficient of variation (CV).
    
    CV = (standard deviation / mean) × 100
    
    This is a key metric for glycemic variability.
    Target: <36% per International Consensus (Battelino 2019).
    Excellent: <33%.
    
    Args:
        values: Array of values.
    
    Returns:
        CV as a percentage (0-100+).
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0 or np.mean(values) == 0:
        return 0.0
    
    return (np.std(values, ddof=1) / np.mean(values)) * 100


def calculate_quantiles(
    values: Union[np.ndarray, pd.Series],
    quantiles: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]
) -> Dict[str, float]:
    """Calculate multiple quantiles for a dataset.
    
    Args:
        values: Array of values.
        quantiles: List of quantiles to calculate (0-1).
    
    Returns:
        Dictionary mapping quantile names (e.g., 'p10', 'p50') to values.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return {f"p{int(q*100)}": None for q in quantiles}
    
    return {
        f"p{int(q*100)}": float(np.percentile(values, q * 100))
        for q in quantiles
    }


def remove_outliers_iqr(
    series: pd.Series,
    factor: float = 1.5,
    quantile_low: float = 0.25,
    quantile_high: float = 0.75
) -> pd.Series:
    """Remove outliers using IQR (Interquartile Range) method.
    
    Values outside [Q1 - factor*IQR, Q3 + factor*IQR] are removed.
    
    Args:
        series: Input series.
        factor: Multiplier for IQR (default 1.5 is standard).
        quantile_low: Lower quantile for IQR calculation.
        quantile_high: Upper quantile for IQR calculation.
    
    Returns:
        Series with outliers removed (as NaN).
    """
    q1 = series.quantile(quantile_low)
    q3 = series.quantile(quantile_high)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    result = series.copy()
    result[(series < lower_bound) | (series > upper_bound)] = np.nan
    
    return result


def remove_outliers_iqr_grouped(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    factor: float = 1.5
) -> pd.DataFrame:
    """Remove outliers using IQR method, computed per group.
    
    Used in cgm_plot.py to remove outliers per minute-of-day.
    
    Args:
        df: DataFrame with data.
        value_column: Column containing values to filter.
        group_column: Column to group by.
        factor: IQR multiplier.
    
    Returns:
        DataFrame with outliers removed.
    """
    def filter_group(group):
        q1 = group[value_column].quantile(0.25)
        q3 = group[value_column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return group[(group[value_column] >= lower) & (group[value_column] <= upper)]
    
    return df.groupby(group_column, group_keys=False).apply(filter_group)


def linear_regression(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series]
) -> Tuple[float, float, float]:
    """Perform simple linear regression.
    
    Args:
        x: Independent variable values.
        y: Dependent variable values.
    
    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    
    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0:
        return 0.0, np.mean(y), 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    # R-squared
    ss_tot = sum_y2 - (sum_y ** 2) / n
    ss_res = sum_y2 - intercept * sum_y - slope * sum_xy
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return float(slope), float(intercept), float(max(0, r_squared))


def calculate_auc(
    values: Union[np.ndarray, pd.Series],
    timestamps: Optional[pd.Series] = None,
    baseline: float = 0.0
) -> float:
    """Calculate Area Under Curve using trapezoidal rule.
    
    Args:
        values: Y-axis values.
        timestamps: Optional timestamps for x-axis (uses index if None).
        baseline: Baseline value to subtract (for incremental AUC).
    
    Returns:
        AUC value.
    """
    values = np.asarray(values) - baseline
    
    if timestamps is not None:
        # Convert to hours for meaningful units
        if hasattr(timestamps, 'dt'):
            # pandas datetime
            x = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600
        else:
            x = np.arange(len(values))
        x = np.asarray(x)
    else:
        x = np.arange(len(values))
    
    # Remove NaN pairs
    mask = ~np.isnan(values)
    x, values = x[mask], values[mask]
    
    if len(values) < 2:
        return 0.0
    
    return float(np.trapz(values, x))


def calculate_auc_above_threshold(
    values: Union[np.ndarray, pd.Series],
    threshold: float,
    timestamps: Optional[pd.Series] = None
) -> float:
    """Calculate AUC for values above a threshold.
    
    Only counts the area where values exceed the threshold.
    Used for "glucose exposure" calculations.
    
    Args:
        values: Y-axis values.
        threshold: Only count area above this value.
        timestamps: Optional timestamps for x-axis.
    
    Returns:
        AUC above threshold.
    """
    values = np.asarray(values)
    above = np.maximum(values - threshold, 0)
    return calculate_auc(above, timestamps, baseline=0.0)


def calculate_time_in_range(
    values: Union[np.ndarray, pd.Series],
    lower: float,
    upper: float
) -> float:
    """Calculate percentage of time values are within a range.
    
    Args:
        values: Array of values.
        lower: Lower bound (inclusive).
        upper: Upper bound (inclusive).
    
    Returns:
        Percentage (0-100) of values within range.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return 0.0
    
    in_range = np.sum((values >= lower) & (values <= upper))
    return (in_range / len(values)) * 100


def calculate_data_coverage(
    df: pd.DataFrame,
    timestamp_column: str = 'timestamp',
    expected_interval_minutes: int = 5
) -> float:
    """Calculate data coverage as percentage of expected readings.
    
    Args:
        df: DataFrame with timestamp column.
        timestamp_column: Name of timestamp column.
        expected_interval_minutes: Expected interval between readings.
    
    Returns:
        Coverage percentage (0-100).
    """
    if len(df) < 2:
        return 0.0
    
    timestamps = df[timestamp_column].sort_values()
    total_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600
    expected_readings = total_hours * (60 / expected_interval_minutes)
    
    if expected_readings == 0:
        return 0.0
    
    return min(100, (len(df) / expected_readings) * 100)
