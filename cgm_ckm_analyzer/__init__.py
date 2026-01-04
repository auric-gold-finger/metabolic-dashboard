"""
Metabolic Dashboard - A comprehensive CGM and ketone data analysis toolkit.

This package provides modular components for:
- Loading data from various CGM and ketone monitor sources
- Analyzing glucose and ketone metrics with evidence-based calculations
- Visualizing metabolic patterns with publication-quality charts
- Generating reports organized by evidence tier
"""

from cgm_ckm_analyzer.config import AnalysisConfig, load_config

__version__ = "2.0.0"
__all__ = ["AnalysisConfig", "load_config"]
