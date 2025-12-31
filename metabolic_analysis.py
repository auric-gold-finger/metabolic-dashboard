"""
Metabolic Data Analysis Module

Ingests and analyzes continuous glucose monitor (CGM) data from Dexcom
and continuous ketone monitor data from Sibio sensors.

Produces comprehensive metabolic analysis including:
- Standard CGM metrics (mean, SD, CV, time-in-range)
- Glucose Management Indicator (GMI)
- Glycemic variability indices
- Ketone analysis and classification
- Combined glucose-ketone metabolic state analysis
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from datetime import datetime, timedelta
import warnings


# =============================================================================
# Data Classes for Analysis Results
# =============================================================================

@dataclass
class GlucoseMetrics:
    """Standard CGM metrics following international consensus guidelines."""
    
    # Basic statistics
    mean: float
    median: float
    std: float
    cv: float  # Coefficient of variation (%)
    
    # Time in range percentages
    time_very_low: float      # <54 mg/dL
    time_low: float           # 54-69 mg/dL
    time_in_range: float      # 70-180 mg/dL
    time_high: float          # 181-250 mg/dL
    time_very_high: float     # >250 mg/dL
    
    # Tighter ranges for metabolic optimization
    time_tight_range: float   # 70-140 mg/dL
    time_optimal: float       # 70-110 mg/dL
    
    # GMI (Glucose Management Indicator)
    gmi: float  # Estimated A1C equivalent
    
    # Variability metrics
    mage: float  # Mean Amplitude of Glycemic Excursions
    j_index: float
    lbgi: float  # Low Blood Glucose Index
    hbgi: float  # High Blood Glucose Index
    
    # Data quality
    readings_count: int
    data_coverage: float  # Percentage of expected readings
    date_range: tuple[datetime, datetime]
    
    def to_dict(self) -> dict:
        return {
            'mean_mg_dl': round(self.mean, 1),
            'median_mg_dl': round(self.median, 1),
            'std_mg_dl': round(self.std, 1),
            'cv_percent': round(self.cv, 1),
            'time_very_low_pct': round(self.time_very_low, 1),
            'time_low_pct': round(self.time_low, 1),
            'time_in_range_pct': round(self.time_in_range, 1),
            'time_high_pct': round(self.time_high, 1),
            'time_very_high_pct': round(self.time_very_high, 1),
            'time_tight_range_pct': round(self.time_tight_range, 1),
            'time_optimal_pct': round(self.time_optimal, 1),
            'gmi_percent': round(self.gmi, 2),
            'mage_mg_dl': round(self.mage, 1),
            'j_index': round(self.j_index, 1),
            'lbgi': round(self.lbgi, 2),
            'hbgi': round(self.hbgi, 2),
            'readings_count': self.readings_count,
            'data_coverage_pct': round(self.data_coverage, 1),
            'start_date': self.date_range[0].isoformat(),
            'end_date': self.date_range[1].isoformat(),
        }


@dataclass 
class KetoneMetrics:
    """Ketone analysis metrics from continuous ketone monitoring."""
    
    # Basic statistics
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    
    # Time in ketone zones (mmol/L thresholds)
    time_absent: float        # <0.2 - no significant ketones
    time_trace: float         # 0.2-0.5 - trace/light ketosis
    time_light: float         # 0.5-1.0 - light nutritional ketosis
    time_moderate: float      # 1.0-3.0 - moderate ketosis
    time_deep: float          # >3.0 - deep ketosis
    
    # Peak analysis
    peak_ketone: float
    peak_timestamp: Optional[datetime]
    time_above_1: float       # Therapeutic ketosis threshold
    time_above_2: float       # Deep ketosis threshold
    
    # Data quality
    readings_count: int
    data_coverage: float
    date_range: tuple[datetime, datetime]
    
    def to_dict(self) -> dict:
        return {
            'mean_mmol_l': round(self.mean, 2),
            'median_mmol_l': round(self.median, 2),
            'std_mmol_l': round(self.std, 2),
            'min_mmol_l': round(self.min_val, 2),
            'max_mmol_l': round(self.max_val, 2),
            'time_absent_pct': round(self.time_absent, 1),
            'time_trace_pct': round(self.time_trace, 1),
            'time_light_pct': round(self.time_light, 1),
            'time_moderate_pct': round(self.time_moderate, 1),
            'time_deep_pct': round(self.time_deep, 1),
            'peak_mmol_l': round(self.peak_ketone, 2),
            'peak_timestamp': self.peak_timestamp.isoformat() if self.peak_timestamp else None,
            'time_above_1_pct': round(self.time_above_1, 1),
            'time_above_2_pct': round(self.time_above_2, 1),
            'readings_count': self.readings_count,
            'data_coverage_pct': round(self.data_coverage, 1),
            'start_date': self.date_range[0].isoformat(),
            'end_date': self.date_range[1].isoformat(),
        }


@dataclass
class DailyMetrics:
    """Per-day breakdown of glucose and ketone metrics."""
    date: str
    glucose_mean: Optional[float] = None
    glucose_std: Optional[float] = None
    glucose_min: Optional[float] = None
    glucose_max: Optional[float] = None
    glucose_tir: Optional[float] = None
    glucose_readings: int = 0
    ketone_mean: Optional[float] = None
    ketone_max: Optional[float] = None
    ketone_readings: int = 0


@dataclass
class MetabolicAnalysis:
    """Combined metabolic analysis results."""
    glucose_metrics: Optional[GlucoseMetrics] = None
    ketone_metrics: Optional[KetoneMetrics] = None
    daily_metrics: list[DailyMetrics] = field(default_factory=list)
    overlap_analysis: Optional[dict] = None  # Analysis of overlapping time periods


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dexcom_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and parse Dexcom CGM export CSV.
    
    Dexcom exports include various event types. We filter for EGV (Estimated 
    Glucose Value) readings which are the actual continuous glucose measurements.
    
    Args:
        filepath: Path to Dexcom CSV export
        
    Returns:
        DataFrame with columns: timestamp, glucose_mg_dl
    """
    # Read with UTF-8-BOM handling
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    
    # Filter for EGV (glucose readings) only
    df = df[df['Event Type'] == 'EGV'].copy()
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)'])
    
    # Extract glucose value
    df['glucose_mg_dl'] = pd.to_numeric(df['Glucose Value (mg/dL)'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=['timestamp', 'glucose_mg_dl'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle duplicate timestamps (take the mean)
    df = df.groupby('timestamp').agg({
        'glucose_mg_dl': 'mean'
    }).reset_index()
    
    return df[['timestamp', 'glucose_mg_dl']]


def load_sibio_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and parse Sibio continuous ketone monitor CSV export.
    
    Args:
        filepath: Path to Sibio CSV export
        
    Returns:
        DataFrame with columns: timestamp, ketone_mmol_l
    """
    # Read CSV - Sibio has some trailing commas creating empty columns
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['Time'])
    
    # Extract ketone value
    df['ketone_mmol_l'] = pd.to_numeric(df['Sensor reading(mmol/L)'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=['timestamp', 'ketone_mmol_l'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df[['timestamp', 'ketone_mmol_l']]


# =============================================================================
# Glucose Analysis Functions
# =============================================================================

def calculate_gmi(mean_glucose: float) -> float:
    """
    Calculate Glucose Management Indicator (GMI) from mean glucose.
    
    GMI estimates HbA1c based on CGM mean glucose using the formula from
    Bergenstal et al. 2018.
    
    Args:
        mean_glucose: Mean glucose in mg/dL
        
    Returns:
        Estimated HbA1c percentage
    """
    return 3.31 + (0.02392 * mean_glucose)


def calculate_mage(glucose_values: np.ndarray, threshold_sd: float = 1.0) -> float:
    """
    Calculate Mean Amplitude of Glycemic Excursions (MAGE).
    
    MAGE measures the average magnitude of glucose excursions that exceed
    one standard deviation from the mean.
    
    Args:
        glucose_values: Array of glucose values
        threshold_sd: Number of SDs to use as excursion threshold (default 1.0)
        
    Returns:
        MAGE value in mg/dL
    """
    if len(glucose_values) < 10:
        return 0.0
    
    sd = np.std(glucose_values)
    threshold = sd * threshold_sd
    
    # Find peaks and nadirs
    excursions = []
    
    # Simple peak/nadir detection
    for i in range(1, len(glucose_values) - 1):
        prev_val = glucose_values[i - 1]
        curr_val = glucose_values[i]
        next_val = glucose_values[i + 1]
        
        # Peak
        if curr_val > prev_val and curr_val > next_val:
            # Find subsequent nadir
            nadir_val = curr_val
            for j in range(i + 1, min(i + 50, len(glucose_values))):
                if glucose_values[j] < nadir_val:
                    nadir_val = glucose_values[j]
                elif glucose_values[j] > nadir_val + threshold / 2:
                    break
            
            excursion = curr_val - nadir_val
            if excursion > threshold:
                excursions.append(excursion)
        
        # Nadir
        elif curr_val < prev_val and curr_val < next_val:
            # Find subsequent peak
            peak_val = curr_val
            for j in range(i + 1, min(i + 50, len(glucose_values))):
                if glucose_values[j] > peak_val:
                    peak_val = glucose_values[j]
                elif glucose_values[j] < peak_val - threshold / 2:
                    break
            
            excursion = peak_val - curr_val
            if excursion > threshold:
                excursions.append(excursion)
    
    return np.mean(excursions) if excursions else 0.0


def calculate_j_index(mean_glucose: float, std_glucose: float) -> float:
    """
    Calculate J-Index for glycemic control assessment.
    
    J-index combines mean glucose and variability into a single metric.
    Lower values indicate better control.
    
    Args:
        mean_glucose: Mean glucose in mg/dL
        std_glucose: Standard deviation of glucose in mg/dL
        
    Returns:
        J-Index value
    """
    return 0.001 * (mean_glucose + std_glucose) ** 2


def calculate_bgri(glucose_values: np.ndarray) -> tuple[float, float]:
    """
    Calculate Blood Glucose Risk Index (LBGI and HBGI).
    
    LBGI/HBGI quantify risk of hypoglycemia and hyperglycemia respectively.
    Based on Kovatchev et al.
    
    Args:
        glucose_values: Array of glucose values in mg/dL
        
    Returns:
        Tuple of (LBGI, HBGI)
    """
    # Transform glucose to symmetric scale
    # f(BG) = 1.509 * [(ln(BG))^1.084 - 5.381]
    glucose_values = np.clip(glucose_values, 20, 600)  # Reasonable bounds
    
    f_bg = 1.509 * (np.power(np.log(glucose_values), 1.084) - 5.381)
    
    # Risk function
    r_bg = 10 * np.power(f_bg, 2)
    
    # Split into low and high risk
    rl_bg = np.where(f_bg < 0, r_bg, 0)
    rh_bg = np.where(f_bg > 0, r_bg, 0)
    
    lbgi = np.mean(rl_bg)
    hbgi = np.mean(rh_bg)
    
    return lbgi, hbgi


def analyze_glucose(df: pd.DataFrame) -> GlucoseMetrics:
    """
    Perform comprehensive glucose analysis following international consensus.
    
    Args:
        df: DataFrame with 'timestamp' and 'glucose_mg_dl' columns
        
    Returns:
        GlucoseMetrics dataclass with all computed metrics
    """
    values = df['glucose_mg_dl'].values
    
    # Basic statistics
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    cv = (std / mean) * 100 if mean > 0 else 0
    
    n = len(values)
    
    # Time in ranges (as percentages)
    time_very_low = (np.sum(values < 54) / n) * 100
    time_low = (np.sum((values >= 54) & (values < 70)) / n) * 100
    time_in_range = (np.sum((values >= 70) & (values <= 180)) / n) * 100
    time_high = (np.sum((values > 180) & (values <= 250)) / n) * 100
    time_very_high = (np.sum(values > 250) / n) * 100
    
    # Tighter ranges for optimization
    time_tight_range = (np.sum((values >= 70) & (values <= 140)) / n) * 100
    time_optimal = (np.sum((values >= 70) & (values <= 110)) / n) * 100
    
    # GMI
    gmi = calculate_gmi(mean)
    
    # Variability metrics
    mage = calculate_mage(values)
    j_index = calculate_j_index(mean, std)
    lbgi, hbgi = calculate_bgri(values)
    
    # Data quality
    date_range = (df['timestamp'].min().to_pydatetime(), 
                  df['timestamp'].max().to_pydatetime())
    
    # Calculate expected readings (5-min intervals)
    total_hours = (date_range[1] - date_range[0]).total_seconds() / 3600
    expected_readings = total_hours * 12  # 12 readings per hour
    data_coverage = (n / expected_readings) * 100 if expected_readings > 0 else 0
    
    return GlucoseMetrics(
        mean=mean,
        median=median,
        std=std,
        cv=cv,
        time_very_low=time_very_low,
        time_low=time_low,
        time_in_range=time_in_range,
        time_high=time_high,
        time_very_high=time_very_high,
        time_tight_range=time_tight_range,
        time_optimal=time_optimal,
        gmi=gmi,
        mage=mage,
        j_index=j_index,
        lbgi=lbgi,
        hbgi=hbgi,
        readings_count=n,
        data_coverage=data_coverage,
        date_range=date_range
    )


# =============================================================================
# Ketone Analysis Functions
# =============================================================================

def analyze_ketones(df: pd.DataFrame) -> KetoneMetrics:
    """
    Perform comprehensive ketone analysis.
    
    Ketone zones:
    - <0.2 mmol/L: Absent (no significant ketosis)
    - 0.2-0.5 mmol/L: Trace (light/fed state)
    - 0.5-1.0 mmol/L: Light nutritional ketosis
    - 1.0-3.0 mmol/L: Moderate ketosis (therapeutic range)
    - >3.0 mmol/L: Deep ketosis
    
    Args:
        df: DataFrame with 'timestamp' and 'ketone_mmol_l' columns
        
    Returns:
        KetoneMetrics dataclass with all computed metrics
    """
    values = df['ketone_mmol_l'].values
    n = len(values)
    
    # Basic statistics
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Time in ketone zones (as percentages)
    time_absent = (np.sum(values < 0.2) / n) * 100
    time_trace = (np.sum((values >= 0.2) & (values < 0.5)) / n) * 100
    time_light = (np.sum((values >= 0.5) & (values < 1.0)) / n) * 100
    time_moderate = (np.sum((values >= 1.0) & (values <= 3.0)) / n) * 100
    time_deep = (np.sum(values > 3.0) / n) * 100
    
    # Therapeutic thresholds
    time_above_1 = (np.sum(values >= 1.0) / n) * 100
    time_above_2 = (np.sum(values >= 2.0) / n) * 100
    
    # Peak analysis
    peak_idx = np.argmax(values)
    peak_ketone = values[peak_idx]
    peak_timestamp = df['timestamp'].iloc[peak_idx].to_pydatetime()
    
    # Data quality
    date_range = (df['timestamp'].min().to_pydatetime(),
                  df['timestamp'].max().to_pydatetime())
    
    total_hours = (date_range[1] - date_range[0]).total_seconds() / 3600
    expected_readings = total_hours * 12
    data_coverage = (n / expected_readings) * 100 if expected_readings > 0 else 0
    
    return KetoneMetrics(
        mean=mean,
        median=median,
        std=std,
        min_val=min_val,
        max_val=max_val,
        time_absent=time_absent,
        time_trace=time_trace,
        time_light=time_light,
        time_moderate=time_moderate,
        time_deep=time_deep,
        peak_ketone=peak_ketone,
        peak_timestamp=peak_timestamp,
        time_above_1=time_above_1,
        time_above_2=time_above_2,
        readings_count=n,
        data_coverage=data_coverage,
        date_range=date_range
    )


# =============================================================================
# Combined Analysis Functions
# =============================================================================

def calculate_daily_metrics(
    glucose_df: Optional[pd.DataFrame],
    ketone_df: Optional[pd.DataFrame]
) -> list[DailyMetrics]:
    """
    Calculate per-day metrics for both glucose and ketones.
    
    Args:
        glucose_df: DataFrame with glucose data (optional)
        ketone_df: DataFrame with ketone data (optional)
        
    Returns:
        List of DailyMetrics for each day with data
    """
    # Collect all unique dates
    dates = set()
    
    if glucose_df is not None:
        glucose_df['date'] = glucose_df['timestamp'].dt.date
        dates.update(glucose_df['date'].unique())
    
    if ketone_df is not None:
        ketone_df['date'] = ketone_df['timestamp'].dt.date
        dates.update(ketone_df['date'].unique())
    
    daily_metrics = []
    
    for date in sorted(dates):
        dm = DailyMetrics(date=str(date))
        
        if glucose_df is not None:
            day_glucose = glucose_df[glucose_df['date'] == date]['glucose_mg_dl']
            if len(day_glucose) > 0:
                dm.glucose_mean = day_glucose.mean()
                dm.glucose_std = day_glucose.std()
                dm.glucose_min = day_glucose.min()
                dm.glucose_max = day_glucose.max()
                # Time in range for the day
                n = len(day_glucose)
                dm.glucose_tir = (np.sum((day_glucose >= 70) & (day_glucose <= 180)) / n) * 100
                dm.glucose_readings = n
        
        if ketone_df is not None:
            day_ketone = ketone_df[ketone_df['date'] == date]['ketone_mmol_l']
            if len(day_ketone) > 0:
                dm.ketone_mean = day_ketone.mean()
                dm.ketone_max = day_ketone.max()
                dm.ketone_readings = len(day_ketone)
        
        daily_metrics.append(dm)
    
    return daily_metrics


def analyze_overlap_periods(
    glucose_df: pd.DataFrame,
    ketone_df: pd.DataFrame,
    time_window_minutes: int = 5
) -> dict:
    """
    Analyze metabolic state during periods with both glucose and ketone data.
    
    Identifies periods of:
    - Optimal metabolic flexibility (low glucose + elevated ketones)
    - Fed state (elevated glucose + low ketones)
    - Stress response (elevated glucose + elevated ketones)
    - Fasted/low-carb (moderate glucose + moderate ketones)
    
    Args:
        glucose_df: DataFrame with glucose data
        ketone_df: DataFrame with ketone data
        time_window_minutes: Window for matching timestamps
        
    Returns:
        Dictionary with overlap analysis results
    """
    # Merge on nearest timestamp
    glucose_df = glucose_df.copy()
    ketone_df = ketone_df.copy()
    
    glucose_df = glucose_df.sort_values('timestamp')
    ketone_df = ketone_df.sort_values('timestamp')
    
    # Use pandas merge_asof for nearest timestamp matching
    merged = pd.merge_asof(
        glucose_df,
        ketone_df,
        on='timestamp',
        tolerance=pd.Timedelta(minutes=time_window_minutes),
        direction='nearest'
    )
    
    # Drop rows where we couldn't match
    merged = merged.dropna(subset=['ketone_mmol_l'])
    
    if len(merged) == 0:
        return {
            'overlap_readings': 0,
            'overlap_hours': 0,
            'metabolic_states': {},
            'message': 'No overlapping time periods found between glucose and ketone data'
        }
    
    n = len(merged)
    
    # Classify metabolic states
    states = {
        'optimal_flexibility': 0,  # glucose <100 + ketones >0.5
        'fed_state': 0,            # glucose >120 + ketones <0.3
        'stress_response': 0,      # glucose >140 + ketones >1.0
        'fasted_keto': 0,          # glucose <100 + ketones 0.5-3.0
        'moderate_mixed': 0,       # everything else
    }
    
    for _, row in merged.iterrows():
        g = row['glucose_mg_dl']
        k = row['ketone_mmol_l']
        
        if g < 100 and k > 0.5:
            if k <= 3.0:
                states['fasted_keto'] += 1
            states['optimal_flexibility'] += 1
        elif g > 120 and k < 0.3:
            states['fed_state'] += 1
        elif g > 140 and k > 1.0:
            states['stress_response'] += 1
        else:
            states['moderate_mixed'] += 1
    
    # Convert to percentages
    state_percentages = {k: (v / n) * 100 for k, v in states.items()}
    
    # Correlation analysis
    correlation = merged['glucose_mg_dl'].corr(merged['ketone_mmol_l'])
    
    # Calculate overlap period
    overlap_hours = (merged['timestamp'].max() - merged['timestamp'].min()).total_seconds() / 3600
    
    return {
        'overlap_readings': n,
        'overlap_hours': round(overlap_hours, 1),
        'metabolic_states_pct': {k: round(v, 1) for k, v in state_percentages.items()},
        'glucose_ketone_correlation': round(correlation, 3),
        'mean_glucose_when_ketones_elevated': round(
            merged[merged['ketone_mmol_l'] >= 0.5]['glucose_mg_dl'].mean(), 1
        ) if len(merged[merged['ketone_mmol_l'] >= 0.5]) > 0 else None,
        'mean_ketones_when_glucose_low': round(
            merged[merged['glucose_mg_dl'] < 100]['ketone_mmol_l'].mean(), 2
        ) if len(merged[merged['glucose_mg_dl'] < 100]) > 0 else None,
    }


# =============================================================================
# Hourly Pattern Analysis
# =============================================================================

def analyze_hourly_patterns(
    glucose_df: Optional[pd.DataFrame] = None,
    ketone_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Analyze hour-of-day patterns for glucose and ketones.
    
    Returns:
        Dictionary with hourly averages and patterns
    """
    result = {}
    
    if glucose_df is not None:
        glucose_df = glucose_df.copy()
        glucose_df['hour'] = glucose_df['timestamp'].dt.hour
        hourly_glucose = glucose_df.groupby('hour')['glucose_mg_dl'].agg(['mean', 'std']).round(1)
        result['glucose_by_hour'] = hourly_glucose.to_dict()
        
        # Find peak and nadir hours
        result['glucose_peak_hour'] = int(hourly_glucose['mean'].idxmax())
        result['glucose_nadir_hour'] = int(hourly_glucose['mean'].idxmin())
        result['dawn_phenomenon'] = round(
            hourly_glucose.loc[6:9, 'mean'].mean() - hourly_glucose.loc[2:4, 'mean'].mean(), 1
        )
    
    if ketone_df is not None:
        ketone_df = ketone_df.copy()
        ketone_df['hour'] = ketone_df['timestamp'].dt.hour
        hourly_ketone = ketone_df.groupby('hour')['ketone_mmol_l'].agg(['mean', 'std']).round(2)
        result['ketone_by_hour'] = hourly_ketone.to_dict()
        
        result['ketone_peak_hour'] = int(hourly_ketone['mean'].idxmax())
        result['ketone_nadir_hour'] = int(hourly_ketone['mean'].idxmin())
    
    return result


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_analysis(
    dexcom_path: Optional[str | Path] = None,
    sibio_path: Optional[str | Path] = None,
) -> MetabolicAnalysis:
    """
    Run complete metabolic analysis on provided data files.
    
    Args:
        dexcom_path: Path to Dexcom CSV export (optional)
        sibio_path: Path to Sibio CSV export (optional)
        
    Returns:
        MetabolicAnalysis dataclass with all results
    """
    result = MetabolicAnalysis()
    glucose_df = None
    ketone_df = None
    
    # Load and analyze glucose data
    if dexcom_path is not None:
        glucose_df = load_dexcom_data(dexcom_path)
        if len(glucose_df) > 0:
            result.glucose_metrics = analyze_glucose(glucose_df)
    
    # Load and analyze ketone data
    if sibio_path is not None:
        ketone_df = load_sibio_data(sibio_path)
        if len(ketone_df) > 0:
            result.ketone_metrics = analyze_ketones(ketone_df)
    
    # Daily metrics
    if glucose_df is not None or ketone_df is not None:
        result.daily_metrics = calculate_daily_metrics(glucose_df, ketone_df)
    
    # Overlap analysis
    if glucose_df is not None and ketone_df is not None:
        result.overlap_analysis = analyze_overlap_periods(glucose_df, ketone_df)
    
    return result


def generate_report(analysis: MetabolicAnalysis) -> str:
    """
    Generate a text report from analysis results.
    
    Args:
        analysis: MetabolicAnalysis results
        
    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("METABOLIC ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    if analysis.glucose_metrics:
        gm = analysis.glucose_metrics
        lines.append("GLUCOSE METRICS (Dexcom CGM)")
        lines.append("-" * 40)
        lines.append(f"Date Range: {gm.date_range[0].strftime('%Y-%m-%d')} to {gm.date_range[1].strftime('%Y-%m-%d')}")
        lines.append(f"Readings: {gm.readings_count:,} ({gm.data_coverage:.1f}% coverage)")
        lines.append("")
        lines.append("Basic Statistics:")
        lines.append(f"  Mean Glucose:     {gm.mean:.1f} mg/dL")
        lines.append(f"  Median Glucose:   {gm.median:.1f} mg/dL")
        lines.append(f"  Std Deviation:    {gm.std:.1f} mg/dL")
        lines.append(f"  CV:               {gm.cv:.1f}%")
        lines.append(f"  GMI (est. A1C):   {gm.gmi:.2f}%")
        lines.append("")
        lines.append("Time in Range:")
        lines.append(f"  Very Low (<54):   {gm.time_very_low:.1f}%")
        lines.append(f"  Low (54-69):      {gm.time_low:.1f}%")
        lines.append(f"  In Range (70-180): {gm.time_in_range:.1f}%")
        lines.append(f"  High (181-250):   {gm.time_high:.1f}%")
        lines.append(f"  Very High (>250): {gm.time_very_high:.1f}%")
        lines.append("")
        lines.append("Optimization Targets:")
        lines.append(f"  Tight Range (70-140): {gm.time_tight_range:.1f}%")
        lines.append(f"  Optimal (70-110):     {gm.time_optimal:.1f}%")
        lines.append("")
        lines.append("Variability Indices:")
        lines.append(f"  MAGE:             {gm.mage:.1f} mg/dL")
        lines.append(f"  J-Index:          {gm.j_index:.1f}")
        lines.append(f"  LBGI:             {gm.lbgi:.2f}")
        lines.append(f"  HBGI:             {gm.hbgi:.2f}")
        lines.append("")
    
    if analysis.ketone_metrics:
        km = analysis.ketone_metrics
        lines.append("KETONE METRICS (Sibio CKM)")
        lines.append("-" * 40)
        lines.append(f"Date Range: {km.date_range[0].strftime('%Y-%m-%d')} to {km.date_range[1].strftime('%Y-%m-%d')}")
        lines.append(f"Readings: {km.readings_count:,} ({km.data_coverage:.1f}% coverage)")
        lines.append("")
        lines.append("Basic Statistics:")
        lines.append(f"  Mean Ketones:     {km.mean:.2f} mmol/L")
        lines.append(f"  Median Ketones:   {km.median:.2f} mmol/L")
        lines.append(f"  Std Deviation:    {km.std:.2f} mmol/L")
        lines.append(f"  Range:            {km.min_val:.2f} - {km.max_val:.2f} mmol/L")
        lines.append(f"  Peak:             {km.peak_ketone:.2f} mmol/L at {km.peak_timestamp.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("Time in Ketone Zones:")
        lines.append(f"  Absent (<0.2):    {km.time_absent:.1f}%")
        lines.append(f"  Trace (0.2-0.5):  {km.time_trace:.1f}%")
        lines.append(f"  Light (0.5-1.0):  {km.time_light:.1f}%")
        lines.append(f"  Moderate (1-3):   {km.time_moderate:.1f}%")
        lines.append(f"  Deep (>3):        {km.time_deep:.1f}%")
        lines.append("")
        lines.append("Therapeutic Thresholds:")
        lines.append(f"  Time ≥1.0 mmol/L: {km.time_above_1:.1f}%")
        lines.append(f"  Time ≥2.0 mmol/L: {km.time_above_2:.1f}%")
        lines.append("")
    
    if analysis.overlap_analysis:
        oa = analysis.overlap_analysis
        lines.append("COMBINED METABOLIC ANALYSIS")
        lines.append("-" * 40)
        if oa.get('overlap_readings', 0) > 0:
            lines.append(f"Overlapping Readings: {oa['overlap_readings']:,}")
            lines.append(f"Overlap Period: {oa['overlap_hours']:.1f} hours")
            lines.append(f"Glucose-Ketone Correlation: {oa['glucose_ketone_correlation']:.3f}")
            lines.append("")
            lines.append("Metabolic State Distribution:")
            for state, pct in oa.get('metabolic_states_pct', {}).items():
                lines.append(f"  {state.replace('_', ' ').title()}: {pct:.1f}%")
            
            if oa.get('mean_glucose_when_ketones_elevated'):
                lines.append(f"\nMean glucose when ketones ≥0.5: {oa['mean_glucose_when_ketones_elevated']:.1f} mg/dL")
            if oa.get('mean_ketones_when_glucose_low'):
                lines.append(f"Mean ketones when glucose <100: {oa['mean_ketones_when_glucose_low']:.2f} mmol/L")
        else:
            lines.append(oa.get('message', 'No overlap data available'))
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for metabolic analysis."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='Analyze CGM glucose and ketone data'
    )
    parser.add_argument(
        '--dexcom', '-g',
        type=str,
        help='Path to Dexcom CSV export'
    )
    parser.add_argument(
        '--sibio', '-k',
        type=str,
        help='Path to Sibio CSV export'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--save', '-s',
        type=str,
        help='Save output to file'
    )
    
    args = parser.parse_args()
    
    if not args.dexcom and not args.sibio:
        parser.error("At least one data file (--dexcom or --sibio) is required")
    
    # Run analysis
    analysis = run_analysis(
        dexcom_path=args.dexcom,
        sibio_path=args.sibio
    )
    
    # Generate output
    if args.output == 'json':
        output = {
            'glucose_metrics': analysis.glucose_metrics.to_dict() if analysis.glucose_metrics else None,
            'ketone_metrics': analysis.ketone_metrics.to_dict() if analysis.ketone_metrics else None,
            'overlap_analysis': analysis.overlap_analysis,
            'daily_metrics': [
                {
                    'date': dm.date,
                    'glucose_mean': round(dm.glucose_mean, 1) if dm.glucose_mean else None,
                    'glucose_std': round(dm.glucose_std, 1) if dm.glucose_std else None,
                    'glucose_tir': round(dm.glucose_tir, 1) if dm.glucose_tir else None,
                    'ketone_mean': round(dm.ketone_mean, 2) if dm.ketone_mean else None,
                    'ketone_max': round(dm.ketone_max, 2) if dm.ketone_max else None,
                }
                for dm in analysis.daily_metrics
            ]
        }
        output_str = json.dumps(output, indent=2)
    else:
        output_str = generate_report(analysis)
    
    # Output
    if args.save:
        with open(args.save, 'w') as f:
            f.write(output_str)
        print(f"Output saved to {args.save}")
    else:
        print(output_str)


if __name__ == '__main__':
    main()
