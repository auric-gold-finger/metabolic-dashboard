"""
Daily metrics dataclass for per-day breakdown.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DailyMetrics:
    """Per-day breakdown of glucose and ketone metrics.
    
    Used for trend analysis and day-over-day comparisons.
    """
    date: str
    
    # Glucose metrics for the day
    glucose_mean: Optional[float] = None
    glucose_std: Optional[float] = None
    glucose_min: Optional[float] = None
    glucose_max: Optional[float] = None
    glucose_cv: Optional[float] = None
    glucose_tir: Optional[float] = None  # Time in range (70-180)
    glucose_readings: int = 0
    
    # Ketone metrics for the day
    ketone_mean: Optional[float] = None
    ketone_min: Optional[float] = None
    ketone_max: Optional[float] = None
    ketone_readings: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date,
            'glucose_mean_mg_dl': round(self.glucose_mean, 1) if self.glucose_mean else None,
            'glucose_std_mg_dl': round(self.glucose_std, 1) if self.glucose_std else None,
            'glucose_min_mg_dl': round(self.glucose_min, 1) if self.glucose_min else None,
            'glucose_max_mg_dl': round(self.glucose_max, 1) if self.glucose_max else None,
            'glucose_cv_pct': round(self.glucose_cv, 1) if self.glucose_cv else None,
            'glucose_tir_pct': round(self.glucose_tir, 1) if self.glucose_tir else None,
            'glucose_readings': self.glucose_readings,
            'ketone_mean_mmol_l': round(self.ketone_mean, 2) if self.ketone_mean else None,
            'ketone_min_mmol_l': round(self.ketone_min, 2) if self.ketone_min else None,
            'ketone_max_mmol_l': round(self.ketone_max, 2) if self.ketone_max else None,
            'ketone_readings': self.ketone_readings,
        }
    
    @property
    def has_glucose(self) -> bool:
        """Check if day has glucose data."""
        return self.glucose_readings > 0
    
    @property
    def has_ketone(self) -> bool:
        """Check if day has ketone data."""
        return self.ketone_readings > 0
