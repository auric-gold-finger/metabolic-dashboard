"""
Ketone metrics dataclass.

Evidence Tier: Optimization (ketogenic diet literature)
Note: Thresholds are from Volek/Phinney and similar sources, not clinically standardized.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


@dataclass
class KetoneMetrics:
    """Ketone analysis metrics from continuous ketone monitoring.
    
    Evidence Tier: Optimization
    
    Ketone zone thresholds are from ketogenic diet literature (Volek/Phinney)
    and are NOT clinically standardized. Thresholds vary by source.
    
    Zones:
    - <0.2 mmol/L: Absent (no significant ketones)
    - 0.2-0.5 mmol/L: Trace (transitional)
    - 0.5-1.0 mmol/L: Light nutritional ketosis
    - 1.0-3.0 mmol/L: Moderate/therapeutic ketosis
    - >3.0 mmol/L: Deep ketosis (approach DKA concern zone for T1D)
    """
    
    # Basic statistics
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    
    # Time in ketone zones (as percentages)
    time_absent: float        # <0.2 mmol/L
    time_trace: float         # 0.2-0.5 mmol/L
    time_light: float         # 0.5-1.0 mmol/L
    time_moderate: float      # 1.0-3.0 mmol/L
    time_deep: float          # >3.0 mmol/L
    
    # Peak analysis
    peak_ketone: float
    peak_timestamp: Optional[datetime]
    time_above_1: float       # Time above therapeutic threshold (1.0)
    time_above_2: float       # Time in deep ketosis (2.0)
    
    # Data quality
    readings_count: int
    data_coverage: float
    date_range: Tuple[datetime, datetime]
    
    @staticmethod
    def get_evidence_tiers() -> Dict[str, str]:
        """Return evidence tier for each metric."""
        # All ketone metrics are optimization tier since thresholds aren't standardized
        return {
            'mean': 'optimization',
            'median': 'optimization',
            'std': 'optimization',
            'time_absent': 'optimization',
            'time_trace': 'optimization',
            'time_light': 'optimization',
            'time_moderate': 'optimization',
            'time_deep': 'optimization',
            'time_above_1': 'optimization',
            'time_above_2': 'optimization',
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
    
    def get_primary_zone(self) -> str:
        """Return the ketone zone with most time spent."""
        zones = {
            'absent': self.time_absent,
            'trace': self.time_trace,
            'light': self.time_light,
            'moderate': self.time_moderate,
            'deep': self.time_deep,
        }
        return max(zones, key=zones.get)
