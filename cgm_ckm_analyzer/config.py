"""
Configuration management for Metabolic Dashboard.

This module provides dataclasses for all configurable thresholds and settings,
with support for loading from YAML files and runtime modification via UI.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class GlucoseThresholds:
    """Glucose thresholds in mg/dL.
    
    Evidence Tiers:
    - Consensus (ADA/EASD): very_low, low, target_low, target_high, very_high
    - Optimization (metabolic health literature): tight_high, optimal_high
    """
    # Consensus thresholds (ADA/EASD/International Consensus)
    very_low: float = 54      # Level 2 hypoglycemia - clinically significant
    low: float = 70           # Level 1 hypoglycemia
    target_low: float = 70    # Standard TIR lower bound
    target_high: float = 180  # Standard TIR upper bound
    high: float = 180         # Elevated glucose
    very_high: float = 250    # Clinically significant hyperglycemia
    
    # Optimization thresholds (non-standard, metabolic health focused)
    tight_low: float = 70     # Same as target_low for tight range
    tight_high: float = 140   # Tighter control target (Attia, Means)
    optimal_low: float = 70   # Optimal/longevity target lower bound
    optimal_high: float = 110 # Optimal/longevity target upper bound


@dataclass
class KetoneThresholds:
    """Ketone thresholds in mmol/L.
    
    Evidence Tier: Optimization (ketogenic diet literature, Volek/Phinney)
    Note: These are not clinically standardized; thresholds vary by source.
    """
    absent: float = 0.2           # Effectively no ketones
    trace: float = 0.5            # Trace ketones, transitional
    light_ketosis: float = 0.5    # Lower bound of nutritional ketosis
    moderate_ketosis: float = 1.0 # Moderate nutritional ketosis
    deep_ketosis: float = 3.0     # Deep ketosis (approaching DKA concern for T1D)
    therapeutic: float = 1.0      # Therapeutic ketosis lower bound


@dataclass
class AnalysisSettings:
    """Settings for analysis algorithms."""
    # CV target - Consensus (Battelino 2019)
    cv_target: float = 36.0
    cv_excellent: float = 33.0
    
    # Readings frequency
    readings_per_hour: int = 12  # 5-minute intervals
    min_data_coverage: float = 70.0  # Minimum % for reliable analysis
    
    # Merge tolerance for aligning glucose/ketone timestamps
    merge_tolerance_minutes: int = 5
    
    # MAGE calculation
    mage_threshold_sd: float = 1.0
    
    # Overnight/dawn phenomenon analysis
    night_start_hour: int = 0
    night_end_hour: int = 6
    dawn_early_start: int = 3
    dawn_early_end: int = 4
    dawn_late_start: int = 5
    dawn_late_end: int = 6


@dataclass
class ExperimentalSettings:
    """Settings for experimental/unvalidated analyses.
    
    Evidence Tier: Experimental
    These thresholds are arbitrary and not clinically validated.
    """
    # Fasting detection (novel algorithm)
    fasting_glucose_threshold: float = 95.0
    fasting_ketone_threshold: float = 0.5
    fasting_min_duration_hours: float = 4.0
    
    # Spike detection
    spike_threshold_rise: float = 30.0  # mg/dL
    spike_roc_threshold: float = 15.0   # mg/dL per hour
    
    # Lag correlation analysis
    max_lag_hours: int = 12
    lag_interval_minutes: int = 30
    
    # Metabolic flexibility score weights (arbitrary, not validated)
    flexibility_glucose_weight: float = 40.0
    flexibility_ketone_weight: float = 30.0
    flexibility_correlation_weight: float = 30.0


@dataclass
class VisualizationSettings:
    """Settings for chart rendering."""
    # Smoothing
    rolling_window: int = 5
    savgol_window: int = 111  # Must be odd
    savgol_polyorder: int = 2
    
    # Font
    font_family: str = "Avenir"
    fallback_font: str = "Inter, sans-serif"
    
    # Quantiles for cgm_plot-style visualization
    quantiles: tuple = (0.10, 0.34, 0.50, 0.68, 0.90)
    
    # IQR outlier removal
    iqr_factor: float = 1.5


@dataclass
class AnalysisConfig:
    """Master configuration container."""
    glucose: GlucoseThresholds = field(default_factory=GlucoseThresholds)
    ketones: KetoneThresholds = field(default_factory=KetoneThresholds)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    experimental: ExperimentalSettings = field(default_factory=ExperimentalSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        """Create config from dictionary."""
        return cls(
            glucose=GlucoseThresholds(**data.get('glucose', {})),
            ketones=KetoneThresholds(**data.get('ketones', {})),
            analysis=AnalysisSettings(**data.get('analysis', {})),
            experimental=ExperimentalSettings(**data.get('experimental', {})),
            visualization=VisualizationSettings(**data.get('visualization', {})),
        )


def load_config(config_path: Optional[Path] = None) -> AnalysisConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml
                    in the cgm_ckm_analyzer package directory.
    
    Returns:
        AnalysisConfig with values from file merged with defaults.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        return AnalysisConfig()
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    # Create config with defaults, then override with file values
    config = AnalysisConfig()
    
    if 'glucose' in data:
        for key, value in data['glucose'].items():
            if hasattr(config.glucose, key):
                setattr(config.glucose, key, value)
    
    if 'ketones' in data:
        for key, value in data['ketones'].items():
            if hasattr(config.ketones, key):
                setattr(config.ketones, key, value)
    
    if 'analysis' in data:
        for key, value in data['analysis'].items():
            if hasattr(config.analysis, key):
                setattr(config.analysis, key, value)
    
    if 'experimental' in data:
        for key, value in data['experimental'].items():
            if hasattr(config.experimental, key):
                setattr(config.experimental, key, value)
    
    if 'visualization' in data:
        for key, value in data['visualization'].items():
            if hasattr(config.visualization, key):
                setattr(config.visualization, key, value)
    
    return config


def save_config(config: AnalysisConfig, config_path: Path) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
