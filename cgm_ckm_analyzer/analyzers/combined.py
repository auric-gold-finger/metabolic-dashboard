"""
Combined Analyzer - Joint glucose and ketone analysis.

Contains analyses that require both glucose and ketone data,
including metabolic state classification, lag correlation,
metabolic flexibility scoring, and fasting detection.

Evidence Tier: EXPERIMENTAL
Most analyses in this module are novel/unvalidated composites.
Use for pattern recognition and exploration only.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from cgm_ckm_analyzer.config import AnalysisConfig
from cgm_ckm_analyzer.analyzers.glucose import GlucoseAnalyzer
from cgm_ckm_analyzer.analyzers.ketone import KetoneAnalyzer
from cgm_ckm_analyzer.metrics.daily_metrics import DailyMetrics
from cgm_ckm_analyzer.utils.statistics import linear_regression


@dataclass
class MetabolicFlexibilityScore:
    """Composite metabolic flexibility score.
    
    EXPERIMENTAL: This is a novel, unvalidated metric.
    
    The score attempts to quantify "metabolic flexibility" - the ability
    to switch between glucose and fat/ketone oxidation - based on:
    
    1. Glucose Stability (0-40 points):
       - Lower CV → more stable glucose regulation
       - Higher TIR → better glucose control
       Formula: (40 - CV) × 2 + TIR × 0.2, capped at 40
    
    2. Ketone Production (0-30 points):
       - Time in ketosis (>0.5 mmol/L) × 0.5, capped at 30
       - Assumes ability to produce ketones indicates fat oxidation capacity
    
    3. Flexibility (0-30 points):
       - Inverse glucose-ketone correlation × 30
       - When glucose drops, ketones should rise (metabolic switching)
       - Strong negative correlation = good flexibility
    
    ORIGIN: This scoring formula was developed for this tool. It is NOT
    published or validated. The weightings (40/30/30) are arbitrary.
    
    INTERPRETATION: Higher scores suggest better metabolic flexibility.
    Missing data reduces the maximum possible score proportionally.
    """
    glucose_stability: Optional[float]
    ketone_production: Optional[float]
    flexibility: Optional[float]
    total: Optional[float]
    max_possible: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'glucose_stability': self.glucose_stability,
            'ketone_production': self.ketone_production,
            'flexibility': self.flexibility,
            'total': self.total,
            'max_possible': self.max_possible,
            'percentage': round((self.total / self.max_possible) * 100, 1) if self.total and self.max_possible else None,
        }


class CombinedAnalyzer:
    """Analyzer for combined glucose and ketone patterns.
    
    Evidence Tier: EXPERIMENTAL
    
    Most analyses in this class are novel composites without published
    validation. They should be used for pattern recognition and exploration,
    not for clinical decisions.
    """
    
    def __init__(
        self,
        glucose_analyzer: GlucoseAnalyzer,
        ketone_analyzer: KetoneAnalyzer,
        config: Optional[AnalysisConfig] = None
    ):
        """Initialize combined analyzer.
        
        Args:
            glucose_analyzer: GlucoseAnalyzer instance.
            ketone_analyzer: KetoneAnalyzer instance.
            config: Optional configuration. Uses defaults if None.
        """
        self.glucose = glucose_analyzer
        self.ketone = ketone_analyzer
        self.config = config or AnalysisConfig()
        self._merged_df: Optional[pd.DataFrame] = None
    
    @property
    def merged_df(self) -> pd.DataFrame:
        """Get merged glucose/ketone DataFrame (lazy load)."""
        if self._merged_df is None:
            self._merged_df = self._merge_data()
        return self._merged_df
    
    def _merge_data(
        self,
        tolerance_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """Merge glucose and ketone data by timestamp.
        
        Args:
            tolerance_minutes: Max time gap for matching.
        
        Returns:
            Merged DataFrame with both glucose and ketone values.
        """
        if tolerance_minutes is None:
            tolerance_minutes = self.config.analysis.merge_tolerance_minutes
        
        glucose_df = self.glucose.df.copy()
        ketone_df = self.ketone.df.copy()
        
        # Use merge_asof for nearest timestamp matching
        merged = pd.merge_asof(
            glucose_df.sort_values('timestamp'),
            ketone_df.sort_values('timestamp'),
            on='timestamp',
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
            direction='nearest'
        )
        
        # Keep only rows with both values
        merged = merged.dropna(subset=['ketone_mmol_l'])
        
        return merged
    
    # =========================================================================
    # OVERLAP ANALYSIS (EXPERIMENTAL)
    # =========================================================================
    
    def analyze_overlap(self) -> Dict[str, Any]:
        """Analyze metabolic state during overlapping data periods.
        
        Evidence Tier: EXPERIMENTAL
        
        Classifies metabolic states based on glucose/ketone combinations:
        - Optimal Flexibility: glucose <100 + ketones >0.5
        - Fed State: glucose >120 + ketones <0.3
        - Stress Response: glucose >140 + ketones >1.0
        - Fasted/Keto: glucose <100 + ketones 0.5-3.0
        
        WARNING: These classifications use arbitrary thresholds and are
        not clinically validated. Labels like "Stress Response" are
        interpretive, not diagnostic.
        
        Returns:
            Dictionary with overlap analysis results.
        """
        merged = self.merged_df
        
        if len(merged) == 0:
            return {
                'overlap_readings': 0,
                'overlap_hours': 0,
                'message': 'No overlapping time periods found',
            }
        
        n = len(merged)
        
        # Classify metabolic states (EXPERIMENTAL thresholds)
        states = {
            'optimal_flexibility': 0,
            'fed_state': 0,
            'stress_response': 0,
            'fasted_keto': 0,
            'moderate_mixed': 0,
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
        state_pct = {k: round((v / n) * 100, 1) for k, v in states.items()}
        
        # Correlation
        correlation = merged['glucose_mg_dl'].corr(merged['ketone_mmol_l'])
        
        # Duration
        overlap_hours = (
            merged['timestamp'].max() - merged['timestamp'].min()
        ).total_seconds() / 3600
        
        return {
            'overlap_readings': n,
            'overlap_hours': round(overlap_hours, 1),
            'metabolic_states_pct': state_pct,
            'glucose_ketone_correlation': round(correlation, 3) if not np.isnan(correlation) else None,
            'mean_glucose_when_ketones_elevated': round(
                merged[merged['ketone_mmol_l'] >= 0.5]['glucose_mg_dl'].mean(), 1
            ) if len(merged[merged['ketone_mmol_l'] >= 0.5]) > 0 else None,
            'mean_ketones_when_glucose_low': round(
                merged[merged['glucose_mg_dl'] < 100]['ketone_mmol_l'].mean(), 2
            ) if len(merged[merged['glucose_mg_dl'] < 100]) > 0 else None,
        }
    
    # =========================================================================
    # LAG CORRELATION (EXPERIMENTAL)
    # =========================================================================
    
    def calculate_lag_correlation(
        self,
        max_lag_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate cross-correlation at various time lags.
        
        Evidence Tier: EXPERIMENTAL
        
        Explores temporal relationship between glucose and ketones.
        A negative correlation at positive lag might indicate ketones
        rising after glucose drops (delayed fat oxidation).
        
        WARNING: This is exploratory analysis. "Optimal lag" is a novel
        concept without validated interpretation.
        
        Args:
            max_lag_hours: Maximum lag to analyze in hours.
        
        Returns:
            Dictionary with lag correlation results.
        """
        if max_lag_hours is None:
            max_lag_hours = self.config.experimental.max_lag_hours
        
        merged = self.merged_df
        
        if len(merged) < 20:
            return {
                'correlations': [],
                'lags': [],
                'optimal_lag': None,
                'max_correlation': None,
                'message': 'Insufficient overlap for lag analysis',
            }
        
        # Calculate correlations at different lags (every 30 min)
        correlations = []
        interval = self.config.experimental.lag_interval_minutes // 5  # Convert to readings
        lags = list(range(-max_lag_hours * 12, max_lag_hours * 12 + 1, interval))
        
        for lag in lags:
            if lag == 0:
                corr = merged['glucose_mg_dl'].corr(merged['ketone_mmol_l'])
            else:
                shifted = merged['ketone_mmol_l'].shift(lag)
                corr = merged['glucose_mg_dl'].corr(shifted)
            correlations.append(corr if not np.isnan(corr) else 0)
        
        lag_hours = [l / 12 for l in lags]  # Convert to hours
        
        # Find strongest inverse correlation
        max_idx = np.argmin(correlations)
        
        return {
            'correlations': correlations,
            'lags': lag_hours,
            'optimal_lag': lag_hours[max_idx],
            'max_correlation': correlations[max_idx],
        }
    
    # =========================================================================
    # METABOLIC FLEXIBILITY SCORE (EXPERIMENTAL)
    # =========================================================================
    
    def calculate_metabolic_flexibility_score(self) -> MetabolicFlexibilityScore:
        """Calculate composite metabolic flexibility score.
        
        Evidence Tier: EXPERIMENTAL
        
        See MetabolicFlexibilityScore docstring for full methodology.
        
        WARNING: This is a novel, unvalidated metric with arbitrary
        weightings. Use for personal tracking only, not clinical decisions.
        
        Returns:
            MetabolicFlexibilityScore dataclass.
        """
        exp = self.config.experimental
        scores = {}
        
        # Glucose stability score (0-40 points)
        glucose_metrics = self.glucose.metrics
        if glucose_metrics:
            cv = glucose_metrics.cv
            tir = glucose_metrics.time_in_range
            
            # Lower CV = better stability
            cv_score = max(0, min(exp.flexibility_glucose_weight, (40 - cv) * 2))
            tir_score = tir * 0.2  # Up to 20 additional points
            glucose_score = min(exp.flexibility_glucose_weight, cv_score + tir_score)
            scores['glucose_stability'] = round(glucose_score, 1)
        else:
            scores['glucose_stability'] = None
        
        # Ketone production score (0-30 points)
        ketone_metrics = self.ketone.metrics
        if ketone_metrics:
            ketosis_time = (
                ketone_metrics.time_light +
                ketone_metrics.time_moderate +
                ketone_metrics.time_deep
            )
            ketone_score = min(exp.flexibility_ketone_weight, ketosis_time * 0.5)
            scores['ketone_production'] = round(ketone_score, 1)
        else:
            scores['ketone_production'] = None
        
        # Flexibility score (0-30 points) - inverse correlation
        overlap = self.analyze_overlap()
        if overlap.get('glucose_ketone_correlation') is not None:
            corr = overlap['glucose_ketone_correlation']
            # Strong negative correlation = good flexibility
            flex_score = max(0, -corr * exp.flexibility_correlation_weight)
            scores['flexibility'] = round(flex_score, 1)
        else:
            scores['flexibility'] = None
        
        # Total score
        valid_scores = [s for s in scores.values() if s is not None]
        total = sum(valid_scores) if valid_scores else None
        
        # Max possible score
        max_possible = sum([
            exp.flexibility_glucose_weight if scores['glucose_stability'] is not None else 0,
            exp.flexibility_ketone_weight if scores['ketone_production'] is not None else 0,
            exp.flexibility_correlation_weight if scores['flexibility'] is not None else 0,
        ])
        
        return MetabolicFlexibilityScore(
            glucose_stability=scores['glucose_stability'],
            ketone_production=scores['ketone_production'],
            flexibility=scores['flexibility'],
            total=round(total, 1) if total is not None else None,
            max_possible=max_possible,
        )
    
    # =========================================================================
    # FASTING DETECTION (EXPERIMENTAL)
    # =========================================================================
    
    def detect_fasting_windows(
        self,
        min_duration_hours: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Detect fasting windows based on glucose and ketone patterns.
        
        Evidence Tier: EXPERIMENTAL
        
        Criteria (default thresholds):
        - Glucose < 95 mg/dL
        - Ketones >= 0.5 mmol/L
        - Duration >= 4 hours
        
        WARNING: There is no standard definition of "fasting" from
        CGM/CKM data. These thresholds are arbitrary approximations.
        
        Returns:
            List of detected fasting windows with start/end times.
        """
        if min_duration_hours is None:
            min_duration_hours = self.config.experimental.fasting_min_duration_hours
        
        exp = self.config.experimental
        merged = self.merged_df.copy()
        
        if len(merged) < 10:
            return []
        
        # Mark fasting periods
        merged['is_fasting'] = (
            (merged['glucose_mg_dl'] < exp.fasting_glucose_threshold) &
            (merged['ketone_mmol_l'] >= exp.fasting_ketone_threshold)
        )
        
        # Find contiguous fasting windows
        merged['fasting_group'] = (
            merged['is_fasting'] != merged['is_fasting'].shift()
        ).cumsum()
        
        fasting_windows = []
        
        for group_id, group_df in merged[merged['is_fasting']].groupby('fasting_group'):
            start = group_df['timestamp'].min()
            end = group_df['timestamp'].max()
            duration_hours = (end - start).total_seconds() / 3600
            
            if duration_hours >= min_duration_hours:
                fasting_windows.append({
                    'start': start.to_pydatetime(),
                    'end': end.to_pydatetime(),
                    'duration_hours': round(duration_hours, 1),
                    'mean_glucose': round(group_df['glucose_mg_dl'].mean(), 1),
                    'mean_ketones': round(group_df['ketone_mmol_l'].mean(), 2),
                    'max_ketones': round(group_df['ketone_mmol_l'].max(), 2),
                })
        
        return fasting_windows
    
    # =========================================================================
    # DAILY METRICS
    # =========================================================================
    
    def calculate_daily_metrics(self) -> List[DailyMetrics]:
        """Calculate per-day metrics for both glucose and ketones.
        
        Evidence Tier: CONSENSUS (standard aggregation)
        
        Returns:
            List of DailyMetrics for each day with data.
        """
        glucose_df = self.glucose.df.copy()
        ketone_df = self.ketone.df.copy()
        
        # Add date column
        glucose_df['date'] = glucose_df['timestamp'].dt.date
        ketone_df['date'] = ketone_df['timestamp'].dt.date
        
        # Collect all unique dates
        dates = set(glucose_df['date'].unique()) | set(ketone_df['date'].unique())
        
        daily_metrics = []
        
        for date in sorted(dates):
            dm = DailyMetrics(date=str(date))
            
            # Glucose metrics for the day
            day_glucose = glucose_df[glucose_df['date'] == date]['glucose_mg_dl']
            if len(day_glucose) > 0:
                dm.glucose_mean = float(day_glucose.mean())
                dm.glucose_std = float(day_glucose.std())
                dm.glucose_min = float(day_glucose.min())
                dm.glucose_max = float(day_glucose.max())
                dm.glucose_cv = (dm.glucose_std / dm.glucose_mean) * 100 if dm.glucose_mean > 0 else 0
                n = len(day_glucose)
                dm.glucose_tir = float((np.sum((day_glucose >= 70) & (day_glucose <= 180)) / n) * 100)
                dm.glucose_readings = n
            
            # Ketone metrics for the day
            day_ketone = ketone_df[ketone_df['date'] == date]['ketone_mmol_l']
            if len(day_ketone) > 0:
                dm.ketone_mean = float(day_ketone.mean())
                dm.ketone_min = float(day_ketone.min())
                dm.ketone_max = float(day_ketone.max())
                dm.ketone_readings = len(day_ketone)
            
            daily_metrics.append(dm)
        
        return daily_metrics
    
    # =========================================================================
    # WEEKLY TRENDS (EXPERIMENTAL)
    # =========================================================================
    
    def analyze_weekly_trends(self) -> Dict[str, Any]:
        """Analyze weekly trends in glucose and ketones.
        
        Evidence Tier: EXPERIMENTAL
        
        Uses linear regression on daily means to detect trends.
        "Improving" interpretation is simplistic.
        
        Returns:
            Dictionary with trend analysis.
        """
        daily = self.calculate_daily_metrics()
        
        if len(daily) < 3:
            return {'message': 'Insufficient data for trend analysis'}
        
        result = {}
        
        # Glucose trend
        glucose_days = [dm for dm in daily if dm.has_glucose]
        if len(glucose_days) >= 3:
            x = np.arange(len(glucose_days))
            y = np.array([dm.glucose_mean for dm in glucose_days])
            slope, intercept, r2 = linear_regression(x, y)
            
            result['glucose_trend'] = {
                'slope': round(slope, 2),
                'r_squared': round(r2, 3),
                'direction': 'decreasing' if slope < -1 else 'increasing' if slope > 1 else 'stable',
                'interpretation': self._interpret_glucose_trend(slope),
            }
        
        # Ketone trend
        ketone_days = [dm for dm in daily if dm.has_ketone]
        if len(ketone_days) >= 3:
            x = np.arange(len(ketone_days))
            y = np.array([dm.ketone_mean for dm in ketone_days])
            slope, intercept, r2 = linear_regression(x, y)
            
            result['ketone_trend'] = {
                'slope': round(slope, 4),
                'r_squared': round(r2, 3),
                'direction': 'decreasing' if slope < -0.01 else 'increasing' if slope > 0.01 else 'stable',
            }
        
        # CV trend (for glucose stability over time)
        cv_days = [dm for dm in glucose_days if dm.glucose_cv is not None]
        if len(cv_days) >= 3:
            x = np.arange(len(cv_days))
            y = np.array([dm.glucose_cv for dm in cv_days])
            slope, intercept, r2 = linear_regression(x, y)
            
            result['cv_trend'] = {
                'slope': round(slope, 2),
                'r_squared': round(r2, 3),
                'direction': 'decreasing' if slope < -0.5 else 'increasing' if slope > 0.5 else 'stable',
                'interpretation': 'improving' if slope < -0.5 else 'worsening' if slope > 0.5 else 'stable',
            }
        
        return result
    
    def _interpret_glucose_trend(self, slope: float) -> str:
        """Interpret glucose trend slope."""
        if slope < -2:
            return 'improving'
        elif slope > 2:
            return 'needs_attention'
        else:
            return 'stable'
