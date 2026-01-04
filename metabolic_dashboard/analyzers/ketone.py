"""
Ketone Analyzer - Core ketone metrics calculation.

Evidence Tier: OPTIMIZATION
Note: Ketone zone thresholds are from ketogenic diet literature and are not
clinically standardized. Thresholds vary by source.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from metabolic_dashboard.config import AnalysisConfig, KetoneThresholds
from metabolic_dashboard.metrics.ketone_metrics import KetoneMetrics
from metabolic_dashboard.utils.statistics import calculate_data_coverage


class KetoneAnalyzer:
    """Analyzer for continuous ketone monitor data.
    
    Computes ketone zone times and patterns.
    
    Evidence Tier: OPTIMIZATION
    
    Ketone zones are from ketogenic diet literature (Volek/Phinney):
    - <0.2 mmol/L: Absent
    - 0.2-0.5 mmol/L: Trace
    - 0.5-1.0 mmol/L: Light nutritional ketosis
    - 1.0-3.0 mmol/L: Moderate/therapeutic ketosis
    - >3.0 mmol/L: Deep ketosis
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[AnalysisConfig] = None
    ):
        """Initialize ketone analyzer.
        
        Args:
            df: DataFrame with 'timestamp' and 'ketone_mmol_l' columns.
            config: Optional configuration. Uses defaults if None.
        """
        self.df = df.copy()
        self.config = config or AnalysisConfig()
        self._metrics: Optional[KetoneMetrics] = None
        self._values: Optional[np.ndarray] = None
        
        # Ensure sorted by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
    
    @property
    def values(self) -> np.ndarray:
        """Get ketone values as numpy array."""
        if self._values is None:
            self._values = self.df['ketone_mmol_l'].values
        return self._values
    
    @property
    def metrics(self) -> KetoneMetrics:
        """Get computed metrics (calculates on first access)."""
        if self._metrics is None:
            self._metrics = self.analyze()
        return self._metrics
    
    # =========================================================================
    # KETONE ZONE CALCULATIONS
    # =========================================================================
    
    def calculate_time_in_zones(self) -> Dict[str, float]:
        """Calculate time in each ketone zone.
        
        Evidence Tier: OPTIMIZATION (ketogenic diet literature)
        
        Returns:
            Dictionary with percentage in each zone.
        """
        t = self.config.ketones
        values = self.values
        n = len(values)
        
        return {
            'absent': (np.sum(values < t.absent) / n) * 100,
            'trace': (np.sum((values >= t.absent) & (values < t.trace)) / n) * 100,
            'light': (np.sum((values >= t.light_ketosis) & (values < t.moderate_ketosis)) / n) * 100,
            'moderate': (np.sum((values >= t.moderate_ketosis) & (values <= t.deep_ketosis)) / n) * 100,
            'deep': (np.sum(values > t.deep_ketosis) / n) * 100,
        }
    
    def calculate_therapeutic_times(self) -> Dict[str, float]:
        """Calculate time above therapeutic thresholds.
        
        Evidence Tier: OPTIMIZATION
        
        Returns:
            Dictionary with percentage above each threshold.
        """
        t = self.config.ketones
        values = self.values
        n = len(values)
        
        return {
            'above_0_5': (np.sum(values >= 0.5) / n) * 100,
            'above_1_0': (np.sum(values >= t.therapeutic) / n) * 100,
            'above_2_0': (np.sum(values >= 2.0) / n) * 100,
            'above_3_0': (np.sum(values >= t.deep_ketosis) / n) * 100,
        }
    
    def find_peak(self) -> Dict[str, Any]:
        """Find peak ketone value and timestamp.
        
        Returns:
            Dictionary with peak value and timestamp.
        """
        peak_idx = np.argmax(self.values)
        return {
            'value': float(self.values[peak_idx]),
            'timestamp': self.df['timestamp'].iloc[peak_idx].to_pydatetime(),
        }
    
    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    
    def analyze_overnight(self) -> Dict[str, Any]:
        """Analyze overnight ketone patterns.
        
        Evidence Tier: OPTIMIZATION
        
        Returns:
            Dictionary with overnight analysis results.
        """
        df = self.df.copy()
        df['hour'] = df['timestamp'].dt.hour
        
        settings = self.config.analysis
        
        # Night window (default 0-6)
        night_mask = (df['hour'] >= settings.night_start_hour) & (df['hour'] < settings.night_end_hour)
        night_values = df.loc[night_mask, 'ketone_mmol_l']
        
        # Morning window (6-9)
        morning_mask = (df['hour'] >= 6) & (df['hour'] < 9)
        morning_values = df.loc[morning_mask, 'ketone_mmol_l']
        
        return {
            'night_mean': float(night_values.mean()) if len(night_values) > 0 else None,
            'night_max': float(night_values.max()) if len(night_values) > 0 else None,
            'morning_mean': float(morning_values.mean()) if len(morning_values) > 0 else None,
            'overnight_rise': (
                float(morning_values.mean() - night_values.mean())
                if len(night_values) > 0 and len(morning_values) > 0
                else None
            ),
        }
    
    def analyze_hourly_patterns(self) -> Dict[str, Any]:
        """Analyze hour-of-day ketone patterns.
        
        Evidence Tier: OPTIMIZATION
        
        Returns:
            Dictionary with hourly statistics.
        """
        df = self.df.copy()
        df['hour'] = df['timestamp'].dt.hour
        
        hourly = df.groupby('hour')['ketone_mmol_l'].agg(['mean', 'std', 'count'])
        
        peak_hour = int(hourly['mean'].idxmax())
        nadir_hour = int(hourly['mean'].idxmin())
        
        return {
            'hourly_mean': hourly['mean'].to_dict(),
            'hourly_std': hourly['std'].to_dict(),
            'hourly_count': hourly['count'].to_dict(),
            'peak_hour': peak_hour,
            'peak_ketone': float(hourly.loc[peak_hour, 'mean']),
            'nadir_hour': nadir_hour,
            'nadir_ketone': float(hourly.loc[nadir_hour, 'mean']),
        }
    
    def get_zone_description(self, value: float) -> str:
        """Get descriptive zone name for a ketone value.
        
        Args:
            value: Ketone value in mmol/L.
        
        Returns:
            Zone description string.
        """
        t = self.config.ketones
        
        if value < t.absent:
            return "Absent (not in ketosis)"
        elif value < t.trace:
            return "Trace (transitional)"
        elif value < t.moderate_ketosis:
            return "Light nutritional ketosis"
        elif value <= t.deep_ketosis:
            return "Moderate/therapeutic ketosis"
        else:
            return "Deep ketosis"
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self) -> KetoneMetrics:
        """Perform comprehensive ketone analysis.
        
        Returns:
            KetoneMetrics with all computed metrics.
        """
        values = self.values
        
        # Basic statistics
        mean = float(np.mean(values))
        median = float(np.median(values))
        std = float(np.std(values, ddof=1))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        # Time in zones
        zones = self.calculate_time_in_zones()
        
        # Therapeutic times
        therapeutic = self.calculate_therapeutic_times()
        
        # Peak
        peak = self.find_peak()
        
        # Data quality
        date_range = (
            self.df['timestamp'].min().to_pydatetime(),
            self.df['timestamp'].max().to_pydatetime()
        )
        data_coverage = calculate_data_coverage(self.df)
        
        return KetoneMetrics(
            mean=mean,
            median=median,
            std=std,
            min_val=min_val,
            max_val=max_val,
            time_absent=zones['absent'],
            time_trace=zones['trace'],
            time_light=zones['light'],
            time_moderate=zones['moderate'],
            time_deep=zones['deep'],
            peak_ketone=peak['value'],
            peak_timestamp=peak['timestamp'],
            time_above_1=therapeutic['above_1_0'],
            time_above_2=therapeutic['above_2_0'],
            readings_count=len(values),
            data_coverage=data_coverage,
            date_range=date_range
        )
