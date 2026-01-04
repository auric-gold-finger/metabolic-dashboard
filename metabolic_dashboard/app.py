"""
Metabolic Analysis Streamlit App (Refactored)

Interactive dashboard for analyzing CGM glucose and ketone monitor data.
Organized by evidence tier: Consensus → Optimization → Experimental.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from metabolic_dashboard.config import AnalysisConfig, load_config
from metabolic_dashboard.loaders import DexcomLoader, SibioLoader
from metabolic_dashboard.analyzers import GlucoseAnalyzer, KetoneAnalyzer, CombinedAnalyzer
from metabolic_dashboard.visualizers import PlotlyVisualizer, MatplotlibVisualizer
from metabolic_dashboard.reports import ReportGenerator


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Metabolic Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .evidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .badge-consensus {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .badge-optimization {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .badge-experimental {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .section-header {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    
    .section-consensus {
        background: linear-gradient(90deg, #dcfce7 0%, transparent 100%);
        border-left: 4px solid #22c55e;
    }
    
    .section-optimization {
        background: linear-gradient(90deg, #fef3c7 0%, transparent 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .section-experimental {
        background: linear-gradient(90deg, #fee2e2 0%, transparent 100%);
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'glucose_df' not in st.session_state:
        st.session_state.glucose_df = None
    if 'ketone_df' not in st.session_state:
        st.session_state.ketone_df = None
    if 'glucose_metrics' not in st.session_state:
        st.session_state.glucose_metrics = None
    if 'ketone_metrics' not in st.session_state:
        st.session_state.ketone_metrics = None


init_session_state()


# =============================================================================
# Sidebar - Configuration
# =============================================================================

def render_sidebar():
    """Render sidebar with file uploads and configuration."""
    with st.sidebar:
        st.title("📊 Metabolic Analysis")
        
        # File uploads
        st.subheader("Data Upload")
        
        glucose_file = st.file_uploader(
            "Dexcom CGM Export (CSV)",
            type=['csv'],
            key='glucose_upload'
        )
        
        ketone_file = st.file_uploader(
            "Sibio Ketone Export (CSV)",
            type=['csv'],
            key='ketone_upload'
        )
        
        # Load data
        if glucose_file is not None:
            try:
                # Save to temp and load
                df = pd.read_csv(glucose_file, encoding='utf-8-sig')
                
                # Filter for EGV if present
                if 'Event Type' in df.columns:
                    df = df[df['Event Type'] == 'EGV'].copy()
                
                # Find columns
                ts_col = next((c for c in df.columns if 'timestamp' in c.lower() or 'time' in c.lower()), None)
                glu_col = next((c for c in df.columns if 'glucose' in c.lower()), None)
                
                if ts_col and glu_col:
                    df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
                    df['glucose_mg_dl'] = pd.to_numeric(df[glu_col], errors='coerce')
                    df = df.dropna(subset=['timestamp', 'glucose_mg_dl'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    st.session_state.glucose_df = df[['timestamp', 'glucose_mg_dl']]
                    st.success(f"Loaded {len(df):,} glucose readings")
            except Exception as e:
                st.error(f"Error loading glucose file: {e}")
        
        if ketone_file is not None:
            try:
                df = pd.read_csv(ketone_file)
                df.columns = df.columns.str.strip()
                
                ts_col = next((c for c in df.columns if 'time' in c.lower()), None)
                ket_col = next((c for c in df.columns if 'mmol' in c.lower() or 'ketone' in c.lower()), None)
                
                if ts_col and ket_col:
                    df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
                    df['ketone_mmol_l'] = pd.to_numeric(df[ket_col], errors='coerce')
                    df = df.dropna(subset=['timestamp', 'ketone_mmol_l'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    st.session_state.ketone_df = df[['timestamp', 'ketone_mmol_l']]
                    st.success(f"Loaded {len(df):,} ketone readings")
            except Exception as e:
                st.error(f"Error loading ketone file: {e}")
        
        st.divider()
        
        # Configuration controls
        st.subheader("⚙️ Settings")
        
        # Smoothing
        smoothing = st.slider("Smoothing Window", 1, 20, 5)
        
        with st.expander("Threshold Settings"):
            config = st.session_state.config
            
            st.caption("Glucose (mg/dL)")
            config.glucose.tight_high = st.number_input(
                "Tight Range Upper",
                value=int(config.glucose.tight_high),
                min_value=100, max_value=200
            )
            config.glucose.optimal_high = st.number_input(
                "Optimal Range Upper",
                value=int(config.glucose.optimal_high),
                min_value=80, max_value=140
            )
            
            st.caption("Analysis")
            config.analysis.cv_target = st.number_input(
                "CV Target (%)",
                value=float(config.analysis.cv_target),
                min_value=20.0, max_value=50.0
            )
            
            if st.button("Reset to Defaults"):
                st.session_state.config = AnalysisConfig()
                st.rerun()
        
        return smoothing


# =============================================================================
# Main Content
# =============================================================================

def render_main():
    """Render main dashboard content."""
    smoothing = render_sidebar()
    
    glucose_df = st.session_state.glucose_df
    ketone_df = st.session_state.ketone_df
    config = st.session_state.config
    
    if glucose_df is None and ketone_df is None:
        st.info("👈 Upload CGM or ketone data files to begin analysis")
        
        with st.expander("ℹ️ About This Dashboard"):
            st.markdown("""
            This dashboard analyzes continuous glucose monitor (CGM) and ketone monitor data,
            organizing metrics by **evidence tier**:
            
            - 🟢 **Consensus Guidelines**: Evidence-based metrics from ADA/EASD/International Consensus
            - 🟡 **Optimization Targets**: Used in metabolic health literature (stricter but less standardized)
            - 🔴 **Experimental Analysis**: Novel/exploratory analyses (use for pattern recognition only)
            
            **Supported Formats:**
            - Dexcom Clarity CSV exports
            - Sibio ketone monitor CSV exports
            """)
        return
    
    # Initialize analyzers
    glucose_analyzer = None
    ketone_analyzer = None
    combined_analyzer = None
    
    if glucose_df is not None:
        glucose_analyzer = GlucoseAnalyzer(glucose_df, config)
    
    if ketone_df is not None:
        ketone_analyzer = KetoneAnalyzer(ketone_df, config)
    
    if glucose_analyzer and ketone_analyzer:
        combined_analyzer = CombinedAnalyzer(glucose_analyzer, ketone_analyzer, config)
    
    # Initialize visualizer (Plotly for interactive charts)
    viz = PlotlyVisualizer(config)
    
    # Create tabs
    tabs = st.tabs(["📊 Overview", "📈 Time Series", "📉 Distribution", "🔬 Advanced"])
    
    # =========================================================================
    # TAB 1: Overview
    # =========================================================================
    with tabs[0]:
        render_overview(glucose_analyzer, ketone_analyzer, combined_analyzer, viz, smoothing)
    
    # =========================================================================
    # TAB 2: Time Series
    # =========================================================================
    with tabs[1]:
        render_time_series(glucose_df, ketone_df, viz, smoothing)
    
    # =========================================================================
    # TAB 3: Distribution
    # =========================================================================
    with tabs[2]:
        render_distribution(glucose_analyzer, ketone_analyzer, viz)
    
    # =========================================================================
    # TAB 4: Advanced
    # =========================================================================
    with tabs[3]:
        render_advanced(glucose_analyzer, ketone_analyzer, combined_analyzer, viz)


def render_overview(glucose_analyzer, ketone_analyzer, combined_analyzer, viz, smoothing):
    """Render overview tab with metrics organized by evidence tier."""
    
    # =========================================================================
    # SECTION 1: Core Metrics (Consensus)
    # =========================================================================
    st.markdown("""
    <div class="section-header section-consensus">
        <strong>📊 Core Metrics</strong>
        <span class="evidence-badge badge-consensus">Consensus Guidelines</span>
        <span style="font-size: 0.85rem; color: #666;">— ADA/EASD/International Consensus</span>
    </div>
    """, unsafe_allow_html=True)
    
    if glucose_analyzer:
        metrics = glucose_analyzer.metrics
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            cv_delta = "✓ At target" if metrics.cv < 36 else "Above target"
            st.metric("CV", f"{metrics.cv:.1f}%", cv_delta)
        
        with col2:
            tir_delta = "✓ At target" if metrics.time_in_range >= 70 else "Below target"
            st.metric("Time in Range", f"{metrics.time_in_range:.1f}%", tir_delta)
        
        with col3:
            st.metric("GMI (est. A1C)", f"{metrics.gmi:.1f}%")
        
        with col4:
            st.metric("Mean Glucose", f"{metrics.mean:.0f} mg/dL")
        
        with col5:
            st.metric("LBGI / HBGI", f"{metrics.lbgi:.1f} / {metrics.hbgi:.1f}")
        
        # TIR breakdown
        with st.expander("Time in Range Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                | Range | % |
                |-------|---|
                | Very Low (<54) | {metrics.time_very_low:.1f}% |
                | Low (54-69) | {metrics.time_low:.1f}% |
                | **In Range (70-180)** | **{metrics.time_in_range:.1f}%** |
                | High (181-250) | {metrics.time_high:.1f}% |
                | Very High (>250) | {metrics.time_very_high:.1f}% |
                """)
            with col2:
                if isinstance(viz, PlotlyVisualizer):
                    fig = viz.create_glucose_tir_donut(metrics)
                    st.plotly_chart(fig, use_container_width=True)
    
    if ketone_analyzer:
        metrics = ketone_analyzer.metrics
        
        st.markdown("**Ketone Summary**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Ketones", f"{metrics.mean:.2f} mmol/L")
        with col2:
            st.metric("Peak", f"{metrics.peak_ketone:.2f} mmol/L")
        with col3:
            st.metric("Time >1.0", f"{metrics.time_above_1:.1f}%")
        with col4:
            st.metric("Readings", f"{metrics.readings_count:,}")
    
    st.divider()
    
    # =========================================================================
    # SECTION 2: Optimization Targets
    # =========================================================================
    st.markdown("""
    <div class="section-header section-optimization">
        <strong>🎯 Optimization Targets</strong>
        <span class="evidence-badge badge-optimization">Metabolic Health</span>
        <span style="font-size: 0.85rem; color: #666;">— Stricter targets from metabolic health literature</span>
    </div>
    """, unsafe_allow_html=True)
    
    if glucose_analyzer:
        metrics = glucose_analyzer.metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tight Range (70-140)", f"{metrics.time_tight_range:.1f}%")
        with col2:
            st.metric("Optimal Range (70-110)", f"{metrics.time_optimal:.1f}%")
        with col3:
            st.metric("MAGE", f"{metrics.mage:.1f} mg/dL")
        with col4:
            st.metric("J-Index", f"{metrics.j_index:.1f}")
    
    if ketone_analyzer:
        metrics = ketone_analyzer.metrics
        
        with st.expander("Ketone Zone Breakdown"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                | Zone | % |
                |------|---|
                | Absent (<0.2) | {metrics.time_absent:.1f}% |
                | Trace (0.2-0.5) | {metrics.time_trace:.1f}% |
                | Light (0.5-1.0) | {metrics.time_light:.1f}% |
                | Moderate (1.0-3.0) | {metrics.time_moderate:.1f}% |
                | Deep (>3.0) | {metrics.time_deep:.1f}% |
                """)
            with col2:
                if isinstance(viz, PlotlyVisualizer):
                    fig = viz.create_ketone_zones_donut(metrics)
                    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # =========================================================================
    # SECTION 3: Experimental Analysis
    # =========================================================================
    st.markdown("""
    <div class="section-header section-experimental">
        <strong>🧪 Experimental Analysis</strong>
        <span class="evidence-badge badge-experimental">Exploratory</span>
        <span style="font-size: 0.85rem; color: #666;">— Novel metrics, use for pattern recognition only</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ These analyses use arbitrary thresholds and are NOT clinically validated.")
    
    if combined_analyzer:
        # Metabolic Flexibility Score
        with st.expander("📊 Metabolic Flexibility Score — How it works"):
            st.markdown("""
            This is a **custom composite score** (0-100) that attempts to quantify "metabolic flexibility":
            
            | Component | Max Points | Calculation | Rationale |
            |-----------|------------|-------------|-----------|
            | **Glucose Stability** | 40 | (40 - CV) × 2 + TIR × 0.2 | Lower variability → more stable |
            | **Ketone Production** | 30 | Time in ketosis × 0.5 | Ability to produce ketones |
            | **Flexibility** | 30 | -correlation × 30 | When glucose ↓, ketones ↑ |
            
            ⚠️ **This scoring formula is NOVEL and UNVALIDATED.** The weightings are arbitrary.
            Use for personal pattern tracking only.
            """)
            
            score = combined_analyzer.calculate_metabolic_flexibility_score()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Glucose Stability", f"{score.glucose_stability or 0:.1f}/40")
            with col2:
                st.metric("Ketone Production", f"{score.ketone_production or 0:.1f}/30")
            with col3:
                st.metric("Flexibility", f"{score.flexibility or 0:.1f}/30")
            with col4:
                st.metric("TOTAL", f"{score.total or 0:.1f}/{score.max_possible:.0f}")
        
        # Overlap Analysis
        overlap = combined_analyzer.analyze_overlap()
        if overlap.get('overlap_readings', 0) > 0:
            with st.expander("🔗 Overlap Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Matched Readings", overlap['overlap_readings'])
                    st.metric("Correlation", f"{overlap.get('glucose_ketone_correlation', 0):.3f}")
                with col2:
                    states = overlap.get('metabolic_states_pct', {})
                    if states:
                        st.markdown("**Metabolic States:**")
                        for state, pct in states.items():
                            if pct > 0:
                                st.text(f"  {state.replace('_', ' ').title()}: {pct:.1f}%")


def render_time_series(glucose_df, ketone_df, viz, smoothing):
    """Render time series tab."""
    st.subheader("📈 Time Series Visualization")
    
    # Always show Plotly interactive charts first
    plotly_viz = PlotlyVisualizer(st.session_state.config)
    
    view_type = st.radio(
        "View Type",
        ['Overlay', 'Stacked'],
        horizontal=True
    )
    
    if view_type == 'Overlay':
        fig = plotly_viz.create_overlay_chart(glucose_df, ketone_df, smoothing=smoothing)
    else:
        fig = plotly_viz.create_time_series_stacked(glucose_df, ketone_df, smoothing=smoothing)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rate of change
    if glucose_df is not None:
        with st.expander("Rate of Change Analysis"):
            fig = plotly_viz.create_rate_of_change_chart(glucose_df, smoothing=smoothing)
            st.plotly_chart(fig, use_container_width=True)
    
    # Rolling variability
    if glucose_df is not None:
        with st.expander("Rolling Variability"):
            window = st.slider("Window (hours)", 6, 48, 24)
            fig = plotly_viz.create_rolling_variability_chart(glucose_df, window_hours=window)
            st.plotly_chart(fig, use_container_width=True)
    
    # Always show Matplotlib publication-quality chart below
    st.divider()
    st.markdown("### 📊 Publication Quality Daily Overlay")
    st.caption("Savitzky-Golay smoothed curves with time-of-day gradient background")
    
    if glucose_df is not None:
        mpl_viz = MatplotlibVisualizer(st.session_state.config)
        mpl_fig = mpl_viz.create_daily_overlay(glucose_df, ketone_df=ketone_df)
        st.pyplot(mpl_fig)
        
        # Download button
        buf = io.BytesIO()
        mpl_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            "📥 Download High-Res PNG",
            buf,
            file_name="cgm_daily_overlay.png",
            mime="image/png"
        )
        plt.close(mpl_fig)  # Clean up


def render_distribution(glucose_analyzer, ketone_analyzer, viz):
    """Render distribution tab."""
    st.subheader("📉 Distribution Analysis")
    
    if isinstance(viz, PlotlyVisualizer):
        col1, col2 = st.columns(2)
        
        if glucose_analyzer:
            with col1:
                st.markdown("**Glucose Distribution**")
                fig = viz.create_histogram(
                    glucose_analyzer.values,
                    title='Glucose Distribution',
                    xlabel='Glucose (mg/dL)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if ketone_analyzer:
            with col2:
                st.markdown("**Ketone Distribution**")
                fig = viz.create_histogram(
                    ketone_analyzer.values,
                    title='Ketone Distribution',
                    xlabel='Ketones (mmol/L)',
                    color='#10b981'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Heatmaps
        st.markdown("**Hourly Patterns**")
        
        col1, col2 = st.columns(2)
        
        if glucose_analyzer:
            with col1:
                fig = viz.create_hourly_heatmap(
                    glucose_analyzer.df,
                    'glucose_mg_dl',
                    'Glucose by Hour & Day'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if ketone_analyzer:
            with col2:
                fig = viz.create_hourly_heatmap(
                    ketone_analyzer.df,
                    'ketone_mmol_l',
                    'Ketones by Hour & Day',
                    colorscale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)


def render_advanced(glucose_analyzer, ketone_analyzer, combined_analyzer, viz):
    """Render advanced analysis tab."""
    st.subheader("🔬 Advanced Analysis")
    
    if combined_analyzer:
        # Lag Correlation
        st.markdown("**Lag Correlation Analysis** (Experimental)")
        st.caption("Explores temporal relationship between glucose and ketones.")
        
        lag_data = combined_analyzer.calculate_lag_correlation()
        if lag_data.get('correlations'):
            if isinstance(viz, PlotlyVisualizer):
                fig = viz.create_lag_correlation_chart(lag_data)
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimal Lag", f"{lag_data['optimal_lag']:.1f} hours")
            with col2:
                st.metric("Max Correlation", f"{lag_data['max_correlation']:.3f}")
        
        st.divider()
        
        # Fasting Windows
        st.markdown("**Fasting Window Detection** (Experimental)")
        st.caption("Detects periods with glucose <95 mg/dL AND ketones ≥0.5 mmol/L for 4+ hours.")
        
        fasting = combined_analyzer.detect_fasting_windows()
        if fasting:
            for i, window in enumerate(fasting[:5]):  # Show first 5
                st.text(f"  Window {i+1}: {window['start'].strftime('%m/%d %H:%M')} - {window['end'].strftime('%H:%M')} ({window['duration_hours']:.1f}h)")
        else:
            st.info("No fasting windows detected with current criteria.")
        
        st.divider()
        
        # Weekly Trends
        st.markdown("**Weekly Trends** (Experimental)")
        trends = combined_analyzer.analyze_weekly_trends()
        
        if 'glucose' in trends:
            if isinstance(viz, PlotlyVisualizer):
                fig = viz.create_trends_chart(trends)
                st.plotly_chart(fig, use_container_width=True)
    
    # Download Report
    st.divider()
    st.markdown("**📄 Export Report**")
    
    if glucose_analyzer or ketone_analyzer:
        generator = ReportGenerator()
        
        report = generator.generate_text_report(
            glucose_metrics=glucose_analyzer.metrics if glucose_analyzer else None,
            ketone_metrics=ketone_analyzer.metrics if ketone_analyzer else None,
            overlap_data=combined_analyzer.analyze_overlap() if combined_analyzer else None,
            flexibility_score=combined_analyzer.calculate_metabolic_flexibility_score().to_dict() if combined_analyzer else None,
        )
        
        st.download_button(
            "Download Text Report",
            report,
            file_name=f"metabolic_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    render_main()
