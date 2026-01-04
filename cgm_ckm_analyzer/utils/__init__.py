"""Utility functions for metabolic analysis."""

from cgm_ckm_analyzer.utils.smoothing import rolling_smooth, savgol_smooth
from cgm_ckm_analyzer.utils.statistics import (
    calculate_cv,
    calculate_quantiles,
    remove_outliers_iqr,
    linear_regression,
)
from cgm_ckm_analyzer.utils.colors import (
    get_glucose_color,
    get_ketone_color,
    GLUCOSE_COLORSCALE,
    KETONE_COLORSCALE,
)

__all__ = [
    "rolling_smooth",
    "savgol_smooth",
    "calculate_cv",
    "calculate_quantiles",
    "remove_outliers_iqr",
    "linear_regression",
    "get_glucose_color",
    "get_ketone_color",
    "GLUCOSE_COLORSCALE",
    "KETONE_COLORSCALE",
]
