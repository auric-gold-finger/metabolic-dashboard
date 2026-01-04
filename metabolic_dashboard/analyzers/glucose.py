"""
Glucose Analyzer - Core glucose metrics calculation.

All calculations follow international consensus guidelines where applicable.
Evidence tier annotations indicate the strength of clinical evidence.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

from metabolic_dashboard.config import AnalysisConfig, GlucoseThresholds
from metabolic_dashboard.metrics.glucose_metrics import GlucoseMetrics
from metabolic_dashboard.utils.statistics import (
    calculate_cv,
    calculate_quantiles,
    calculate_auc_above_threshold,
    calculate_time_in_range,
    calculate_data_coverage,
)


class GlucoseAnalyzer:
    """Analyzer for continuous glucose monitor (CGM) data.
    
    Computes standard CGM metrics following international consensus guidelines
    (Battelino 2019, Bergenstal 2018, Kovatchev).
    
    Metrics are annotated with evidence tiers:
    - Consensus: Established clinical guidelines
    - Optimization: Used in metabolic health literature
    - Experimental: Novel/unvalidated analyses
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[AnalysisConfig] = None
    ):
        """Initialize glucose analyzer.
        
        Args:
            df: DataFrame with 'timestamp' and 'glucose_mg_dl' columns.
            config: Optional configuration. Uses defaults if None.
        """
        self.df = df.copy()
        self.config = config or AnalysisConfig()
        self._metrics: Optional[GlucoseMetrics] = None
        self._values: Optional[np.ndarray] = None
        
        # Ensure sorted by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
    
    @property
    def values(self) -> np.ndarray:
        """Get glucose values as numpy array."""
        if self._values is None:
            self._values = self.df['glucose_mg_dl'].values
        return self._values
    
    @property
    def metrics(self) -> GlucoseMetrics:
        """Get computed metrics (calculates on first access)."""
        if self._metrics is None:
            self._metrics = self.analyze()
        return self._metrics
    
    # =========================================================================
    # CONSENSUS METRICS (International Guidelines)
    # =========================================================================
    
    def calculate_gmi(self, mean_glucose: Optional[float] = None) -> float:
        """Calculate Glucose Management Indicator (GMI).
        
        Evidence Tier: CONSENSUS (Bergenstal et al., 2018, Diabetes Care)
        
        GMI estimates HbA1c from CGM mean glucose using the validated formula.
        
        Args:
            mean_glucose: Mean glucose in mg/dL. Uses data mean if None.
        
        Returns:
            Estimated HbA1c percentage.
        """
        if mean_glucose is None:
            mean_glucose = np.mean(self.values)
        return 3.31 + (0.02392 * mean_glucose)
    
    def calculate_bgri(self) -> Tuple[float, float]:
        """Calculate Blood Glucose Risk Index (LBGI and HBGI).
        
        Evidence Tier: CONSENSUS (Kovatchev et al.)
        
        LBGI/HBGI quantify risk of hypoglycemia and hyperglycemia.
        
        Returns:
            Tuple of (LBGI, HBGI).
        """
        # Clip to reasonable bounds
        values = np.clip(self.values, 20, 600)
        
        # Transform glucose to symmetric scale
        # f(BG) = 1.509 * [(ln(BG))^1.084 - 5.381]
        f_bg = 1.509 * (np.power(np.log(values), 1.084) - 5.381)
        
        # Risk function: r(BG) = 10 * f(BG)^2
        r_bg = 10 * np.power(f_bg, 2)
        
        # Split into low and high risk
        rl_bg = np.where(f_bg < 0, r_bg, 0)
        rh_bg = np.where(f_bg > 0, r_bg, 0)
        
        lbgi = float(np.mean(rl_bg))
        hbgi = float(np.mean(rh_bg))
        
        return lbgi, hbgi
    
    def calculate_time_in_ranges(self) -> Dict[str, float]:
        """Calculate time in standard glucose ranges.
        
        Evidence Tier: CONSENSUS (ADA/EASD, Battelino 2019)
        
        Standard ranges:
        - Very Low: <54 mg/dL (Level 2 hypoglycemia)
        - Low: 54-69 mg/dL (Level 1 hypoglycemia)
        - In Range: 70-180 mg/dL (target >70%)
        - High: 181-250 mg/dL
        - Very High: >250 mg/dL
        
        Returns:
            Dictionary with percentage in each range.
        """
        t = self.config.glucose
        values = self.values
        n = len(values)
        
        return {
            'very_low': (np.sum(values < t.very_low) / n) * 100,
            'low': (np.sum((values >= t.very_low) & (values < t.low)) / n) * 100,
            'in_range': (np.sum((values >= t.target_low) & (values <= t.target_high)) / n) * 100,
            'high': (np.sum((values > t.target_high) & (values <= t.very_high)) / n) * 100,
            'very_high': (np.sum(values > t.very_high) / n) * 100,
        }
    
    # =========================================================================
    # OPTIMIZATION METRICS (Metabolic Health Literature)
    # =========================================================================
    
    def calculate_tight_ranges(self) -> Dict[str, float]:
        """Calculate time in tighter optimization ranges.
        
        Evidence Tier: OPTIMIZATION (Attia, Means, metabolic health literature)
        
        These are stricter than clinical diabetes targets:
        - Tight: 70-140 mg/dL
        - Optimal: 70-110 mg/dL
        
        Returns:
            Dictionary with percentage in each range.
        """
        t = self.config.glucose
        values = self.values
        n = len(values)
        
        return {
            'tight_range': (np.sum((values >= t.target_low) & (values <= t.tight_high)) / n) * 100,
            'optimal': (np.sum((values >= t.optimal_low) & (values <= t.optimal_high)) / n) * 100,
        }
    
    def calculate_mage(self, threshold_sd: Optional[float] = None) -> float:
        """Calculate Mean Amplitude of Glycemic Excursions (MAGE).
        
        Evidence Tier: OPTIMIZATION (Service et al., 1970)
        
        MAGE measures average magnitude of glucose excursions exceeding
        one standard deviation from mean.
        
        Args:
            threshold_sd: SD multiplier for excursion threshold.
        
        Returns:
            MAGE value in mg/dL.
        """
        if threshold_sd is None:
            threshold_sd = self.config.analysis.mage_threshold_sd
        
        values = self.values
        
        if len(values) < 10:
            return 0.0
        
        sd = np.std(values)
        threshold = sd * threshold_sd
        excursions = []
        
        # Simple peak/nadir detection
        for i in range(1, len(values) - 1):
            prev_val = values[i - 1]
            curr_val = values[i]
            next_val = values[i + 1]
            
            # Peak detection
            if curr_val > prev_val and curr_val > next_val:
                # Find subsequent nadir
                nadir_val = curr_val
                for j in range(i + 1, min(i + 50, len(values))):
                    if values[j] < nadir_val:
                        nadir_val = values[j]
                    elif values[j] > nadir_val + threshold / 2:
                        break
                
                excursion = curr_val - nadir_val
                if excursion > threshold:
                    excursions.append(excursion)
            
            # Nadir detection
            elif curr_val < prev_val and curr_val < next_val:
                # Find subsequent peak
                peak_val = curr_val
                for j in range(i + 1, min(i + 50, len(values))):
                    if values[j] > peak_val:
                        peak_val = values[j]
                    elif values[j] < peak_val - threshold / 2:
                        break
                
                excursion = peak_val - curr_val
                if excursion > threshold:
                    excursions.append(excursion)
        
        return float(np.mean(excursions)) if excursions else 0.0
    
    def calculate_j_index(self) -> float:
        """Calculate J-Index for glycemic control.
        
        Evidence Tier: OPTIMIZATION (Wójcicki, 1995)
        
        J-index combines mean and variability: 0.001 × (mean + SD)²
        Lower values indicate better control.
        
        Returns:
            J-Index value.
        """
        mean = np.mean(self.values)
        std = np.std(self.values)
        return 0.001 * (mean + std) ** 2
    
    def calculate_quantiles(
        self,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Calculate glucose percentiles.
        
        Evidence Tier: CONSENSUS (standard statistics)
        
        Args:
            quantiles: List of quantiles (0-1). Defaults to config.
        
        Returns:
            Dictionary mapping percentile names to values.
        """
        if quantiles is None:
            quantiles = list(self.config.visualization.quantiles)
        
        return calculate_quantiles(self.values, quantiles)
    
    def calculate_auc_above(self, threshold: float = 100) -> float:
        """Calculate area under curve above a threshold.
        
        Evidence Tier: OPTIMIZATION
        
        Useful for quantifying glucose exposure above optimal levels.
        
        Args:
            threshold: Glucose threshold in mg/dL.
        
        Returns:
            AUC value (mg/dL × hours).
        """
        return calculate_auc_above_threshold(
            self.values,
            threshold,
            self.df['timestamp']
        )
    
    # =========================================================================
    # RATE OF CHANGE AND VARIABILITY
    # =========================================================================
    
    def calculate_rate_of_change(self) -> pd.Series:
        """Calculate glucose rate of change over time.
        
        Evidence Tier: OPTIMIZATION
        
        Returns:
            Series with rate of change in mg/dL per hour.
        """
        df = self.df.copy()
        
        # Calculate time difference in hours
        df['time_diff_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600
        
        # Calculate glucose difference
        df['glucose_diff'] = df['glucose_mg_dl'].diff()
        
        # Rate of change (mg/dL per hour)
        df['roc'] = df['glucose_diff'] / df['time_diff_hours']
        
        # Clean up extreme values from data gaps
        df.loc[df['time_diff_hours'] > 1, 'roc'] = np.nan
        
        return df['roc']
    
    def calculate_rolling_variability(
        self,
        window_hours: int = 24
    ) -> pd.DataFrame:
        """Calculate rolling CV and SD over time.
        
        Evidence Tier: OPTIMIZATION
        
        Args:
            window_hours: Rolling window size in hours.
        
        Returns:
            DataFrame with rolling CV and SD.
        """
        df = self.df.copy()
        
        # Assume 5-minute readings (12/hour)
        window_size = window_hours * self.config.analysis.readings_per_hour
        
        df['rolling_mean'] = df['glucose_mg_dl'].rolling(
            window=window_size, min_periods=window_size // 2
        ).mean()
        
        df['rolling_std'] = df['glucose_mg_dl'].rolling(
            window=window_size, min_periods=window_size // 2
        ).std()
        
        df['rolling_cv'] = (df['rolling_std'] / df['rolling_mean']) * 100
        
        return df[['timestamp', 'rolling_mean', 'rolling_std', 'rolling_cv']]
    
    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    
    def analyze_overnight(self) -> Dict[str, Any]:
        """Analyze overnight glucose patterns including dawn phenomenon.
        
        Evidence Tier: OPTIMIZATION (dawn phenomenon is well-documented)
        
        Returns:
            Dictionary with overnight analysis results.
        """
        df = self.df.copy()
        df['hour'] = df['timestamp'].dt.hour
        
        settings = self.config.analysis
        
        # Night window (default 0-6)
        night_mask = (df['hour'] >= settings.night_start_hour) & (df['hour'] < settings.night_end_hour)
        night_values = df.loc[night_mask, 'glucose_mg_dl']
        
        # Early morning (3-4 AM) for dawn phenomenon
        early_mask = (df['hour'] >= settings.dawn_early_start) & (df['hour'] < settings.dawn_early_end)
        early_values = df.loc[early_mask, 'glucose_mg_dl']
        
        # Late morning (5-6 AM) for dawn phenomenon
        late_mask = (df['hour'] >= settings.dawn_late_start) & (df['hour'] < settings.dawn_late_end)
        late_values = df.loc[late_mask, 'glucose_mg_dl']
        
        dawn_effect = None
        if len(early_values) > 0 and len(late_values) > 0:
            dawn_effect = float(late_values.mean() - early_values.mean())
        
        return {
            'night_mean': float(night_values.mean()) if len(night_values) > 0 else None,
            'night_std': float(night_values.std()) if len(night_values) > 0 else None,
            'night_min': float(night_values.min()) if len(night_values) > 0 else None,
            'night_max': float(night_values.max()) if len(night_values) > 0 else None,
            'dawn_effect': dawn_effect,
            'dawn_interpretation': self._interpret_dawn_effect(dawn_effect),
        }
    
    def _interpret_dawn_effect(self, dawn_effect: Optional[float]) -> str:
        """Interpret dawn phenomenon magnitude."""
        if dawn_effect is None:
            return 'insufficient_data'
        if dawn_effect < 5:
            return 'minimal'
        elif dawn_effect < 10:
            return 'mild'
        elif dawn_effect < 20:
            return 'moderate'
        else:
            return 'significant'
    
    def analyze_hourly_patterns(self) -> Dict[str, Any]:
        """Analyze hour-of-day glucose patterns.
        
        Evidence Tier: CONSENSUS (standard time-series analysis)
        
        Returns:
            Dictionary with hourly statistics.
        """
        df = self.df.copy()
        df['hour'] = df['timestamp'].dt.hour
        
        hourly = df.groupby('hour')['glucose_mg_dl'].agg(['mean', 'std', 'count'])
        
        peak_hour = int(hourly['mean'].idxmax())
        nadir_hour = int(hourly['mean'].idxmin())
        
        return {
            'hourly_mean': hourly['mean'].to_dict(),
            'hourly_std': hourly['std'].to_dict(),
            'hourly_count': hourly['count'].to_dict(),
            'peak_hour': peak_hour,
            'peak_glucose': float(hourly.loc[peak_hour, 'mean']),
            'nadir_hour': nadir_hour,
            'nadir_glucose': float(hourly.loc[nadir_hour, 'mean']),
        }
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self) -> GlucoseMetrics:
        """Perform comprehensive glucose analysis.
        
        Returns:
            GlucoseMetrics with all computed metrics.
        """
        values = self.values
        
        # Basic statistics
        mean = float(np.mean(values))
        median = float(np.median(values))
        std = float(np.std(values, ddof=1))
        cv = (std / mean) * 100 if mean > 0 else 0
        
        # Time in ranges (consensus)
        tir = self.calculate_time_in_ranges()
        
        # Tight ranges (optimization)
        tight = self.calculate_tight_ranges()
        
        # GMI
        gmi = self.calculate_gmi(mean)
        
        # Variability metrics
        mage = self.calculate_mage()
        j_index = self.calculate_j_index()
        lbgi, hbgi = self.calculate_bgri()
        
        # Data quality
        date_range = (
            self.df['timestamp'].min().to_pydatetime(),
            self.df['timestamp'].max().to_pydatetime()
        )
        data_coverage = calculate_data_coverage(self.df)
        
        return GlucoseMetrics(
            mean=mean,
            median=median,
            std=std,
            cv=cv,
            time_very_low=tir['very_low'],
            time_low=tir['low'],
            time_in_range=tir['in_range'],
            time_high=tir['high'],
            time_very_high=tir['very_high'],
            time_tight_range=tight['tight_range'],
            time_optimal=tight['optimal'],
            gmi=gmi,
            mage=mage,
            j_index=j_index,
            lbgi=lbgi,
            hbgi=hbgi,
            readings_count=len(values),
            data_coverage=data_coverage,
            date_range=date_range
        )
