"""
Glucose metrics dataclass.

Follows international consensus guidelines (Battelino 2019) for CGM metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Dict, Any


@dataclass
class GlucoseMetrics:
    """Standard CGM metrics following international consensus guidelines.
    
    Evidence Tiers:
    - Consensus: mean, median, std, cv, time_very_low, time_low, time_in_range,
                 time_high, time_very_high, gmi, lbgi, hbgi
    - Optimization: time_tight_range, time_optimal
    - Moderate: mage, j_index
    """
    
    # Basic statistics (Consensus)
    mean: float
    median: float
    std: float
    cv: float  # Coefficient of variation (%)
    
    # Time in range percentages (Consensus - ADA/EASD)
    time_very_low: float      # <54 mg/dL - Level 2 hypoglycemia
    time_low: float           # 54-69 mg/dL - Level 1 hypoglycemia
    time_in_range: float      # 70-180 mg/dL - Standard TIR
    time_high: float          # 181-250 mg/dL - Elevated
    time_very_high: float     # >250 mg/dL - Very high
    
    # Tighter ranges for metabolic optimization (Optimization tier)
    time_tight_range: float   # 70-140 mg/dL
    time_optimal: float       # 70-110 mg/dL
    
    # GMI - Glucose Management Indicator (Consensus - Bergenstal 2018)
    gmi: float  # Estimated A1C equivalent
    
    # Variability metrics (Moderate evidence)
    mage: float  # Mean Amplitude of Glycemic Excursions
    j_index: float
    
    # Blood Glucose Risk Index (Consensus - Kovatchev)
    lbgi: float  # Low Blood Glucose Index
    hbgi: float  # High Blood Glucose Index
    
    # Data quality
    readings_count: int
    data_coverage: float  # Percentage of expected readings
    date_range: Tuple[datetime, datetime]
    
    # Evidence tier metadata
    @staticmethod
    def get_evidence_tiers() -> Dict[str, str]:
        """Return evidence tier for each metric."""
        return {
            'mean': 'consensus',
            'median': 'consensus',
            'std': 'consensus',
            'cv': 'consensus',
            'time_very_low': 'consensus',
            'time_low': 'consensus',
            'time_in_range': 'consensus',
            'time_high': 'consensus',
            'time_very_high': 'consensus',
            'gmi': 'consensus',
            'lbgi': 'consensus',
            'hbgi': 'consensus',
            'time_tight_range': 'optimization',
            'time_optimal': 'optimization',
            'mage': 'optimization',
            'j_index': 'optimization',
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
