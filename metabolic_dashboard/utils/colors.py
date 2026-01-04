"""
Color utilities for metabolic data visualization.

Provides color scales and mapping functions for glucose and ketone values.
Includes both the original cgm_plot.py matplotlib palette and Plotly-compatible scales.
"""

import numpy as np
from typing import Tuple, List, Optional
from metabolic_dashboard.config import GlucoseThresholds, KetoneThresholds


# =============================================================================
# GLUCOSE COLOR SCALES
# =============================================================================

# Plotly-compatible colorscale for glucose heatmaps
GLUCOSE_COLORSCALE = [
    [0.0, '#8B0000'],      # Very low (<54) - dark red
    [0.15, '#FF6B6B'],     # Low (54-70) - red
    [0.25, '#FFD93D'],     # Low-normal - yellow
    [0.4, '#6BCB77'],      # In range (70-110) - green
    [0.55, '#4ECDC4'],     # Upper normal (110-140) - cyan
    [0.7, '#9B59B6'],      # Elevated (140-180) - purple
    [0.85, '#FF8C00'],     # High (180-250) - orange
    [1.0, '#8B0000'],      # Very high (>250) - dark red
]

# Ketone colorscale
KETONE_COLORSCALE = [
    [0.0, '#E8E8E8'],      # Absent (<0.2) - light gray
    [0.2, '#B8D4E3'],      # Trace (0.2-0.5) - light blue
    [0.4, '#7EB8DA'],      # Light ketosis (0.5-1.0) - blue
    [0.6, '#4A90A4'],      # Moderate (1.0-2.0) - medium blue
    [0.8, '#2E6B7E'],      # Deep (2.0-3.0) - dark blue
    [1.0, '#1A3A47'],      # Very deep (>3.0) - very dark blue
]


def get_glucose_color(
    value: float,
    thresholds: Optional[GlucoseThresholds] = None
) -> str:
    """Get color for a glucose value based on thresholds.
    
    Args:
        value: Glucose value in mg/dL.
        thresholds: Optional custom thresholds. Uses defaults if None.
    
    Returns:
        Hex color string.
    """
    if thresholds is None:
        thresholds = GlucoseThresholds()
    
    if value < thresholds.very_low:
        return '#8B0000'  # Dark red - severe hypoglycemia
    elif value < thresholds.low:
        return '#FF6B6B'  # Red - hypoglycemia
    elif value < thresholds.optimal_high:  # 70-110
        return '#6BCB77'  # Green - optimal
    elif value < thresholds.tight_high:  # 110-140
        return '#4ECDC4'  # Cyan - good
    elif value < thresholds.target_high:  # 140-180
        return '#9B59B6'  # Purple - elevated
    elif value < thresholds.very_high:  # 180-250
        return '#FF8C00'  # Orange - high
    else:  # >250
        return '#8B0000'  # Dark red - very high


def get_ketone_color(
    value: float,
    thresholds: Optional[KetoneThresholds] = None
) -> str:
    """Get color for a ketone value based on thresholds.
    
    Args:
        value: Ketone value in mmol/L.
        thresholds: Optional custom thresholds. Uses defaults if None.
    
    Returns:
        Hex color string.
    """
    if thresholds is None:
        thresholds = KetoneThresholds()
    
    if value < thresholds.absent:
        return '#E8E8E8'  # Light gray - absent
    elif value < thresholds.trace:
        return '#B8D4E3'  # Light blue - trace
    elif value < thresholds.moderate_ketosis:
        return '#7EB8DA'  # Blue - light ketosis
    elif value < thresholds.deep_ketosis:
        return '#4A90A4'  # Medium blue - moderate ketosis
    else:
        return '#2E6B7E'  # Dark blue - deep ketosis


# =============================================================================
# CGM PLOT MATPLOTLIB PALETTE (from cgm_plot.py)
# =============================================================================

# Time-of-day gradient colors (RGB normalized 0-1)
NIGHT_COLOR = np.array([0, 32, 128]) / 255.0
DAYTIME_COLOR = np.array([255, 223, 186]) / 255.0
SUNRISE_COLOR = np.array([38, 125, 255]) / 255.0  # #267DFF
SUNSET_COLOR = np.array([218, 89, 85]) / 255.0    # #DA5955

# Default CGM plot line color
CGM_LINE_COLOR = '#267DFF'
CGM_MAX_LINE_COLOR = '#FF0000'


def interpolate_colors(
    color1: np.ndarray,
    color2: np.ndarray,
    num_steps: int
) -> List[Tuple[float, float, float]]:
    """Generate interpolated color values between two colors.
    
    Args:
        color1: Starting RGB color (0-1 normalized).
        color2: Ending RGB color (0-1 normalized).
        num_steps: Number of color steps to generate.
    
    Returns:
        List of RGB tuples.
    """
    if num_steps <= 1:
        return [(color1[0], color1[1], color1[2])]
    
    r = np.linspace(color1[0], color2[0], num_steps)
    g = np.linspace(color1[1], color2[1], num_steps)
    b = np.linspace(color1[2], color2[2], num_steps)
    
    return list(zip(r, g, b))


def get_time_of_day_periods() -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Get time-of-day gradient periods for CGM plot background.
    
    Returns:
        List of (start_minute, end_minute, start_color, end_color) tuples.
    """
    return [
        # Night to sunrise (5:00-6:10)
        (300, 370, NIGHT_COLOR, SUNRISE_COLOR),
        # Sunrise to day (6:10-7:20)
        (370, 440, SUNRISE_COLOR, DAYTIME_COLOR),
        # Day (7:20-18:20)
        (440, 1100, DAYTIME_COLOR, DAYTIME_COLOR),
        # Day to sunset (18:20-20:00)
        (1100, 1200, DAYTIME_COLOR, SUNSET_COLOR),
        # Sunset to night (20:00-21:40)
        (1200, 1300, SUNSET_COLOR, NIGHT_COLOR),
        # Night (0:00-5:00)
        (0, 300, NIGHT_COLOR, NIGHT_COLOR),
        # Night (21:40-24:00)
        (1300, 1440, NIGHT_COLOR, NIGHT_COLOR),
    ]


def get_glucose_reference_lines() -> List[Tuple[float, str, str, float]]:
    """Get glucose reference lines for plots.
    
    Returns:
        List of (value, color, linestyle, alpha) tuples.
    """
    return [
        (90, 'grey', '--', 0.5),
        (100, 'green', '-', 0.5),
        (110, 'yellow', '--', 0.5),
        (140, 'red', '--', 0.5),
    ]


# =============================================================================
# EVIDENCE TIER COLORS
# =============================================================================

EVIDENCE_TIER_COLORS = {
    'consensus': '#22C55E',      # Green - well-established
    'optimization': '#F59E0B',   # Amber - moderate evidence
    'experimental': '#EF4444',   # Red - experimental/unvalidated
}

EVIDENCE_TIER_LABELS = {
    'consensus': '🟢 Consensus',
    'optimization': '🟡 Optimization',
    'experimental': '🔴 Experimental',
}


def get_evidence_tier_badge(tier: str) -> str:
    """Get formatted badge text for an evidence tier.
    
    Args:
        tier: One of 'consensus', 'optimization', 'experimental'.
    
    Returns:
        Formatted badge string.
    """
    return EVIDENCE_TIER_LABELS.get(tier, tier)
