"""
Report Generator - Text and structured report generation.

Generates analysis reports with evidence tier annotations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from cgm_ckm_analyzer.config import AnalysisConfig
from cgm_ckm_analyzer.metrics.glucose_metrics import GlucoseMetrics
from cgm_ckm_analyzer.metrics.ketone_metrics import KetoneMetrics
from cgm_ckm_analyzer.metrics.daily_metrics import DailyMetrics
from cgm_ckm_analyzer.utils.colors import EVIDENCE_TIER_LABELS


class ReportGenerator:
    """Generate analysis reports with evidence tier annotations.
    
    Reports are organized into three sections:
    - Core Metrics (Consensus): Evidence-based clinical guidelines
    - Optimization Targets: Metabolic health optimization
    - Experimental Analysis: Novel/unvalidated analyses
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize report generator.
        
        Args:
            config: Optional configuration.
        """
        self.config = config or AnalysisConfig()
    
    def generate_text_report(
        self,
        glucose_metrics: Optional[GlucoseMetrics] = None,
        ketone_metrics: Optional[KetoneMetrics] = None,
        overlap_data: Optional[Dict[str, Any]] = None,
        flexibility_score: Optional[Dict[str, Any]] = None,
        trends: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate full text report.
        
        Args:
            glucose_metrics: Glucose analysis results.
            ketone_metrics: Ketone analysis results.
            overlap_data: Overlap analysis results.
            flexibility_score: Metabolic flexibility score.
            trends: Weekly trend analysis.
        
        Returns:
            Formatted text report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("METABOLIC ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)
        lines.append("")
        
        # Core Metrics Section
        lines.append("-" * 60)
        lines.append("CORE METRICS (Consensus Guidelines)")
        lines.append("-" * 60)
        lines.append("")
        
        if glucose_metrics:
            lines.append("GLUCOSE ANALYSIS")
            lines.append(f"  Date Range: {glucose_metrics.date_range[0].strftime('%Y-%m-%d')} to {glucose_metrics.date_range[1].strftime('%Y-%m-%d')}")
            lines.append(f"  Readings: {glucose_metrics.readings_count:,} ({glucose_metrics.data_coverage:.1f}% coverage)")
            lines.append("")
            lines.append("  Statistics:")
            lines.append(f"    Mean: {glucose_metrics.mean:.1f} mg/dL")
            lines.append(f"    Median: {glucose_metrics.median:.1f} mg/dL")
            lines.append(f"    SD: {glucose_metrics.std:.1f} mg/dL")
            lines.append(f"    CV: {glucose_metrics.cv:.1f}% (target <36%)")
            lines.append("")
            lines.append("  Time in Range (Standard):")
            lines.append(f"    Very Low (<54): {glucose_metrics.time_very_low:.1f}%")
            lines.append(f"    Low (54-69): {glucose_metrics.time_low:.1f}%")
            lines.append(f"    In Range (70-180): {glucose_metrics.time_in_range:.1f}% (target >70%)")
            lines.append(f"    High (181-250): {glucose_metrics.time_high:.1f}%")
            lines.append(f"    Very High (>250): {glucose_metrics.time_very_high:.1f}%")
            lines.append("")
            lines.append("  Key Indicators:")
            lines.append(f"    GMI (est. A1C): {glucose_metrics.gmi:.1f}%")
            lines.append(f"    LBGI (hypo risk): {glucose_metrics.lbgi:.2f}")
            lines.append(f"    HBGI (hyper risk): {glucose_metrics.hbgi:.2f}")
            lines.append("")
        
        if ketone_metrics:
            lines.append("KETONE ANALYSIS")
            lines.append(f"  Readings: {ketone_metrics.readings_count:,}")
            lines.append(f"  Mean: {ketone_metrics.mean:.2f} mmol/L")
            lines.append(f"  Peak: {ketone_metrics.peak_ketone:.2f} mmol/L")
            lines.append("")
        
        # Optimization Section
        lines.append("-" * 60)
        lines.append("OPTIMIZATION TARGETS")
        lines.append("-" * 60)
        lines.append("")
        
        if glucose_metrics:
            lines.append("  Tighter Glucose Ranges:")
            lines.append(f"    Tight (70-140): {glucose_metrics.time_tight_range:.1f}%")
            lines.append(f"    Optimal (70-110): {glucose_metrics.time_optimal:.1f}%")
            lines.append("")
            lines.append("  Variability:")
            lines.append(f"    MAGE: {glucose_metrics.mage:.1f} mg/dL")
            lines.append(f"    J-Index: {glucose_metrics.j_index:.1f}")
            lines.append("")
        
        if ketone_metrics:
            lines.append("  Ketone Zones:")
            lines.append(f"    Absent (<0.2): {ketone_metrics.time_absent:.1f}%")
            lines.append(f"    Trace (0.2-0.5): {ketone_metrics.time_trace:.1f}%")
            lines.append(f"    Light (0.5-1.0): {ketone_metrics.time_light:.1f}%")
            lines.append(f"    Moderate (1.0-3.0): {ketone_metrics.time_moderate:.1f}%")
            lines.append(f"    Deep (>3.0): {ketone_metrics.time_deep:.1f}%")
            lines.append("")
        
        # Experimental Section
        if overlap_data or flexibility_score:
            lines.append("-" * 60)
            lines.append("EXPERIMENTAL ANALYSIS")
            lines.append("   Note: These metrics are exploratory and unvalidated")
            lines.append("-" * 60)
            lines.append("")
            
            if overlap_data and overlap_data.get('overlap_readings', 0) > 0:
                lines.append("  Overlap Analysis:")
                lines.append(f"    Matched readings: {overlap_data['overlap_readings']}")
                lines.append(f"    Correlation: {overlap_data.get('glucose_ketone_correlation', 'N/A')}")
                lines.append("")
            
            if flexibility_score:
                lines.append("  Metabolic Flexibility Score:")
                lines.append(f"    Glucose Stability: {flexibility_score.get('glucose_stability', 'N/A')}/40")
                lines.append(f"    Ketone Production: {flexibility_score.get('ketone_production', 'N/A')}/30")
                lines.append(f"    Flexibility: {flexibility_score.get('flexibility', 'N/A')}/30")
                lines.append(f"    TOTAL: {flexibility_score.get('total', 'N/A')}/{flexibility_score.get('max_possible', 100)}")
                lines.append("")
        
        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def generate_summary_dict(
        self,
        glucose_metrics: Optional[GlucoseMetrics] = None,
        ketone_metrics: Optional[KetoneMetrics] = None,
    ) -> Dict[str, Any]:
        """Generate structured summary dictionary.
        
        Args:
            glucose_metrics: Glucose analysis results.
            ketone_metrics: Ketone analysis results.
        
        Returns:
            Dictionary with summary data.
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'glucose': glucose_metrics.to_dict() if glucose_metrics else None,
            'ketones': ketone_metrics.to_dict() if ketone_metrics else None,
        }
        
        return summary
    
    def get_interpretation(
        self,
        glucose_metrics: Optional[GlucoseMetrics] = None,
        ketone_metrics: Optional[KetoneMetrics] = None,
    ) -> Dict[str, str]:
        """Generate interpretive text for key metrics.
        
        Args:
            glucose_metrics: Glucose analysis results.
            ketone_metrics: Ketone analysis results.
        
        Returns:
            Dictionary with metric interpretations.
        """
        interpretations = {}
        
        if glucose_metrics:
            # CV interpretation
            if glucose_metrics.cv < 33:
                interpretations['cv'] = 'Excellent glycemic stability'
            elif glucose_metrics.cv < 36:
                interpretations['cv'] = 'Good glycemic stability (at target)'
            elif glucose_metrics.cv < 40:
                interpretations['cv'] = 'Moderate variability, room for improvement'
            else:
                interpretations['cv'] = 'High variability, consider interventions'
            
            # TIR interpretation
            if glucose_metrics.time_in_range >= 70:
                interpretations['tir'] = 'Meeting consensus target (>70%)'
            elif glucose_metrics.time_in_range >= 50:
                interpretations['tir'] = 'Below target, focus on reducing highs/lows'
            else:
                interpretations['tir'] = 'Significantly below target'
            
            # GMI interpretation
            if glucose_metrics.gmi < 5.7:
                interpretations['gmi'] = 'Non-diabetic range'
            elif glucose_metrics.gmi < 6.5:
                interpretations['gmi'] = 'Prediabetic range'
            else:
                interpretations['gmi'] = 'Diabetic range'
        
        if ketone_metrics:
            # Ketone interpretation
            if ketone_metrics.mean < 0.2:
                interpretations['ketones'] = 'Not in ketosis'
            elif ketone_metrics.mean < 0.5:
                interpretations['ketones'] = 'Trace ketones'
            elif ketone_metrics.mean < 1.0:
                interpretations['ketones'] = 'Light nutritional ketosis'
            elif ketone_metrics.mean < 2.0:
                interpretations['ketones'] = 'Moderate ketosis'
            else:
                interpretations['ketones'] = 'Deep ketosis'
        
        return interpretations
