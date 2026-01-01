"""
Metabolic Analysis Streamlit App

Interactive dashboard for analyzing CGM glucose and ketone monitor data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

from metabolic_analysis import (
    load_dexcom_data,
    load_sibio_data,
    analyze_glucose,
    analyze_ketones,
    calculate_daily_metrics,
    analyze_overlap_periods,
    analyze_hourly_patterns,
    run_analysis,
    generate_report,
)

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Metabolic Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric cards - use inherit to respect theme */
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
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.15s ease;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    
    [data-testid="stFileUploader"] > div:first-child {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success/Error/Warning/Info messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Number inputs */
    input[type="number"] {
        border-radius: 8px !important;
    }
    
    /* Chart containers - transparent to inherit theme background */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

def get_glucose_color(value: float) -> str:
    """Return color based on glucose value."""
    if value < 54:
        return "#ef4444"  # red - very low
    elif value < 70:
        return "#f59e0b"  # amber - low
    elif value <= 110:
        return "#10b981"  # emerald - optimal
    elif value <= 140:
        return "#06b6d4"  # cyan - good
    elif value <= 180:
        return "#8b5cf6"  # violet - ok
    elif value <= 250:
        return "#f97316"  # orange - high
    else:
        return "#ef4444"  # red - very high


def get_ketone_color(value: float) -> str:
    """Return color based on ketone value."""
    if value < 0.2:
        return "#94a3b8"  # slate - absent
    elif value < 0.5:
        return "#06b6d4"  # cyan - trace
    elif value < 1.0:
        return "#10b981"  # emerald - light
    elif value <= 3.0:
        return "#22c55e"  # green - moderate
    else:
        return "#f59e0b"  # amber - deep


def smooth_data(df: pd.DataFrame, column: str, window: int = 1) -> pd.Series:
    """Apply rolling average smoothing to data.
    
    Args:
        df: DataFrame with the data
        column: Column name to smooth
        window: Window size for rolling average (1 = no smoothing)
    
    Returns:
        Smoothed series
    """
    if window <= 1:
        return df[column]
    return df[column].rolling(window=window, center=True, min_periods=1).mean()


# =============================================================================
# Advanced Analysis Functions
# =============================================================================

def calculate_rate_of_change(df: pd.DataFrame, column: str, time_col: str = 'timestamp') -> pd.Series:
    """Calculate rate of change (velocity) in units per hour."""
    time_diff = df[time_col].diff().dt.total_seconds() / 3600  # hours
    value_diff = df[column].diff()
    roc = value_diff / time_diff
    return roc.fillna(0)


def calculate_rolling_variability(df: pd.DataFrame, column: str, window_hours: int = 24) -> pd.DataFrame:
    """Calculate rolling CV and std deviation over specified window."""
    # Estimate readings per hour (typically 12 for CGM at 5-min intervals)
    readings_per_hour = len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600)
    window_size = max(1, int(window_hours * readings_per_hour))
    
    rolling = df[column].rolling(window=window_size, center=True, min_periods=int(window_size * 0.5))
    
    result = pd.DataFrame({
        'timestamp': df['timestamp'],
        'rolling_mean': rolling.mean(),
        'rolling_std': rolling.std(),
        'rolling_cv': (rolling.std() / rolling.mean()) * 100
    })
    return result


def calculate_lag_correlation(glucose_df: pd.DataFrame, ketone_df: pd.DataFrame, 
                              max_lag_hours: int = 12) -> dict:
    """Calculate cross-correlation between glucose and ketones at different time lags."""
    # Merge on nearest timestamp
    merged = pd.merge_asof(
        glucose_df.sort_values('timestamp'),
        ketone_df.sort_values('timestamp'),
        on='timestamp',
        tolerance=pd.Timedelta(minutes=10),
        direction='nearest'
    ).dropna()
    
    if len(merged) < 20:
        return {'correlations': [], 'lags': [], 'optimal_lag': 0, 'max_correlation': 0}
    
    # Calculate correlation at different lags
    correlations = []
    lags = list(range(-max_lag_hours * 12, max_lag_hours * 12 + 1, 6))  # every 30 min
    
    for lag in lags:
        if lag == 0:
            corr = merged['glucose_mg_dl'].corr(merged['ketone_mmol_l'])
        else:
            shifted = merged['ketone_mmol_l'].shift(lag)
            corr = merged['glucose_mg_dl'].corr(shifted)
        correlations.append(corr if not np.isnan(corr) else 0)
    
    lag_hours = [l / 12 for l in lags]  # Convert to hours
    max_idx = np.argmin(correlations)  # Most negative = strongest inverse correlation
    
    return {
        'correlations': correlations,
        'lags': lag_hours,
        'optimal_lag': lag_hours[max_idx],
        'max_correlation': correlations[max_idx]
    }


def calculate_metabolic_flexibility_score(glucose_metrics, ketone_metrics, overlap_data: dict = None) -> dict:
    """Calculate composite metabolic flexibility score (0-100)."""
    scores = {}
    
    # Glucose stability score (0-40 points)
    # CV < 20% = excellent, CV > 40% = poor
    if glucose_metrics:
        cv = glucose_metrics.cv
        cv_score = max(0, min(40, (40 - cv) * 2)) if cv <= 40 else 0
        tir_score = glucose_metrics.time_in_range * 0.2  # Up to 20 points for TIR
        glucose_score = cv_score + tir_score
        scores['glucose_stability'] = round(glucose_score, 1)
    else:
        scores['glucose_stability'] = None
    
    # Ketone production score (0-30 points)
    if ketone_metrics:
        # Time in nutritional ketosis (>0.5) up to 20 points
        ketosis_time = ketone_metrics.time_light + ketone_metrics.time_moderate + ketone_metrics.time_deep
        ketone_score = min(30, ketosis_time * 0.5)
        scores['ketone_production'] = round(ketone_score, 1)
    else:
        scores['ketone_production'] = None
    
    # Flexibility score (0-30 points) - inverse correlation strength
    if overlap_data and overlap_data.get('glucose_ketone_correlation'):
        # Strong negative correlation = good flexibility
        corr = overlap_data['glucose_ketone_correlation']
        flex_score = max(0, -corr * 30)  # -1 correlation = 30 points
        scores['flexibility'] = round(flex_score, 1)
    else:
        scores['flexibility'] = None
    
    # Total score
    valid_scores = [s for s in scores.values() if s is not None]
    scores['total'] = round(sum(valid_scores), 1) if valid_scores else None
    scores['max_possible'] = sum([40 if scores['glucose_stability'] is not None else 0,
                                   30 if scores['ketone_production'] is not None else 0,
                                   30 if scores['flexibility'] is not None else 0])
    
    return scores


def detect_fasting_windows(glucose_df: pd.DataFrame, ketone_df: pd.DataFrame = None,
                           glucose_threshold: float = 95, ketone_threshold: float = 0.5,
                           min_duration_hours: float = 4) -> list:
    """Detect fasting windows based on glucose stability and ketone elevation."""
    fasting_windows = []
    
    if glucose_df is None or len(glucose_df) < 10:
        return fasting_windows
    
    df = glucose_df.copy()
    df['is_low_glucose'] = df['glucose_mg_dl'] < glucose_threshold
    
    # Add ketone data if available
    if ketone_df is not None:
        merged = pd.merge_asof(
            df.sort_values('timestamp'),
            ketone_df[['timestamp', 'ketone_mmol_l']].sort_values('timestamp'),
            on='timestamp',
            tolerance=pd.Timedelta(minutes=30),
            direction='nearest'
        )
        merged['has_ketones'] = merged['ketone_mmol_l'].fillna(0) >= ketone_threshold
        merged['is_fasting'] = merged['is_low_glucose'] & merged['has_ketones']
    else:
        merged = df
        merged['is_fasting'] = merged['is_low_glucose']
    
    # Find contiguous fasting periods
    merged['fasting_group'] = (merged['is_fasting'] != merged['is_fasting'].shift()).cumsum()
    
    for group_id, group in merged[merged['is_fasting']].groupby('fasting_group'):
        duration = (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 3600
        if duration >= min_duration_hours:
            fasting_windows.append({
                'start': group['timestamp'].min(),
                'end': group['timestamp'].max(),
                'duration_hours': round(duration, 1),
                'avg_glucose': round(group['glucose_mg_dl'].mean(), 1),
                'min_glucose': round(group['glucose_mg_dl'].min(), 1),
                'avg_ketones': round(group['ketone_mmol_l'].mean(), 2) if 'ketone_mmol_l' in group.columns else None
            })
    
    return fasting_windows


def analyze_glucose_spikes(glucose_df: pd.DataFrame, threshold_rise: float = 30,
                           min_duration_minutes: int = 15) -> list:
    """Detect and analyze glucose spikes."""
    if glucose_df is None or len(glucose_df) < 10:
        return []
    
    df = glucose_df.copy().sort_values('timestamp')
    df['roc'] = calculate_rate_of_change(df, 'glucose_mg_dl')
    
    spikes = []
    in_spike = False
    spike_start_idx = None
    baseline = None
    
    for i in range(1, len(df)):
        current = df.iloc[i]['glucose_mg_dl']
        prev = df.iloc[i-1]['glucose_mg_dl']
        
        if not in_spike:
            # Detect spike start (rapid rise)
            if current - prev > 5 and df.iloc[i]['roc'] > 15:  # Rising fast
                in_spike = True
                spike_start_idx = i - 1
                baseline = prev
        else:
            # Detect spike peak (rate of change becomes negative)
            if df.iloc[i]['roc'] < 0:
                # Find peak
                spike_data = df.iloc[spike_start_idx:i+1]
                peak_idx = spike_data['glucose_mg_dl'].idxmax()
                peak_value = spike_data.loc[peak_idx, 'glucose_mg_dl']
                peak_time = spike_data.loc[peak_idx, 'timestamp']
                
                rise = peak_value - baseline
                
                if rise >= threshold_rise:
                    # Calculate time to peak
                    start_time = df.iloc[spike_start_idx]['timestamp']
                    time_to_peak = (peak_time - start_time).total_seconds() / 60
                    
                    # Find recovery (return to within 10% of baseline)
                    recovery_target = baseline + (rise * 0.1)
                    recovery_data = df.iloc[peak_idx:]
                    recovery_idx = recovery_data[recovery_data['glucose_mg_dl'] <= recovery_target].first_valid_index()
                    
                    if recovery_idx is not None:
                        recovery_time = (df.loc[recovery_idx, 'timestamp'] - peak_time).total_seconds() / 60
                    else:
                        recovery_time = None
                    
                    # Calculate AUC (simplified)
                    auc = spike_data['glucose_mg_dl'].sum() - (baseline * len(spike_data))
                    
                    spikes.append({
                        'start_time': start_time,
                        'peak_time': peak_time,
                        'baseline': round(baseline, 1),
                        'peak': round(peak_value, 1),
                        'rise': round(rise, 1),
                        'time_to_peak_min': round(time_to_peak, 1),
                        'recovery_time_min': round(recovery_time, 1) if recovery_time else None,
                        'auc': round(auc, 1)
                    })
                
                in_spike = False
    
    return spikes


def analyze_overnight(glucose_df: pd.DataFrame, ketone_df: pd.DataFrame = None,
                      night_start: int = 0, night_end: int = 6) -> dict:
    """Analyze overnight/sleep period patterns."""
    if glucose_df is None or len(glucose_df) < 10:
        return {}
    
    df = glucose_df.copy()
    df['hour'] = df['timestamp'].dt.hour
    overnight = df[(df['hour'] >= night_start) & (df['hour'] < night_end)]
    
    if len(overnight) < 5:
        return {}
    
    # Basic overnight stats
    result = {
        'overnight_mean': round(overnight['glucose_mg_dl'].mean(), 1),
        'overnight_std': round(overnight['glucose_mg_dl'].std(), 1),
        'overnight_cv': round((overnight['glucose_mg_dl'].std() / overnight['glucose_mg_dl'].mean()) * 100, 1),
        'overnight_min': round(overnight['glucose_mg_dl'].min(), 1),
        'overnight_max': round(overnight['glucose_mg_dl'].max(), 1),
    }
    
    # Dawn phenomenon: compare 3-4am to 5-6am
    early_night = df[(df['hour'] >= 3) & (df['hour'] < 4)]['glucose_mg_dl'].mean()
    dawn = df[(df['hour'] >= 5) & (df['hour'] < 6)]['glucose_mg_dl'].mean()
    
    if not np.isnan(early_night) and not np.isnan(dawn):
        result['dawn_phenomenon'] = round(dawn - early_night, 1)
        result['dawn_phenomenon_pct'] = round(((dawn - early_night) / early_night) * 100, 1)
    
    # Overnight ketone analysis
    if ketone_df is not None:
        kdf = ketone_df.copy()
        kdf['hour'] = kdf['timestamp'].dt.hour
        overnight_k = kdf[(kdf['hour'] >= night_start) & (kdf['hour'] < night_end)]
        
        if len(overnight_k) > 0:
            result['overnight_ketone_mean'] = round(overnight_k['ketone_mmol_l'].mean(), 2)
            result['overnight_ketone_max'] = round(overnight_k['ketone_mmol_l'].max(), 2)
    
    return result


def calculate_weekly_trends(glucose_df: pd.DataFrame, ketone_df: pd.DataFrame = None) -> dict:
    """Calculate weekly trends with linear regression."""
    result = {'glucose': None, 'ketones': None}
    
    if glucose_df is not None and len(glucose_df) > 50:
        df = glucose_df.copy()
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg({
            'glucose_mg_dl': ['mean', 'std']
        }).reset_index()
        daily.columns = ['date', 'mean', 'std']
        daily['cv'] = (daily['std'] / daily['mean']) * 100
        daily['day_num'] = range(len(daily))
        
        if len(daily) >= 3:
            # Linear regression on mean
            slope_mean, intercept_mean = np.polyfit(daily['day_num'], daily['mean'], 1)
            # Linear regression on CV
            slope_cv, intercept_cv = np.polyfit(daily['day_num'], daily['cv'].fillna(0), 1)
            
            result['glucose'] = {
                'dates': [str(d) for d in daily['date'].tolist()],
                'means': daily['mean'].tolist(),
                'cvs': daily['cv'].tolist(),
                'mean_trend_slope': round(slope_mean, 2),
                'cv_trend_slope': round(slope_cv, 2),
                'mean_trend_line': (intercept_mean + slope_mean * daily['day_num']).tolist(),
                'cv_trend_line': (intercept_cv + slope_cv * daily['day_num']).tolist(),
                'improving': slope_mean < 0 and slope_cv < 0  # Lower mean and CV is better
            }
    
    if ketone_df is not None and len(ketone_df) > 50:
        df = ketone_df.copy()
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg({
            'ketone_mmol_l': 'mean'
        }).reset_index()
        daily.columns = ['date', 'mean']
        daily['day_num'] = range(len(daily))
        
        if len(daily) >= 3:
            slope, intercept = np.polyfit(daily['day_num'], daily['mean'], 1)
            
            result['ketones'] = {
                'dates': [str(d) for d in daily['date'].tolist()],
                'means': daily['mean'].tolist(),
                'trend_slope': round(slope, 3),
                'trend_line': (intercept + slope * daily['day_num']).tolist(),
                'improving': slope > 0  # Higher ketones (within reason) is better for ketogenic
            }
    
    return result


def create_rate_of_change_chart(glucose_df: pd.DataFrame, smoothing: int = 5) -> go.Figure:
    """Create glucose rate of change visualization."""
    df = glucose_df.copy()
    df['roc'] = calculate_rate_of_change(df, 'glucose_mg_dl')
    df['roc_smooth'] = smooth_data(df, 'roc', smoothing) if 'roc' in df.columns else df['roc']
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Glucose", "Rate of Change (mg/dL per hour)")
    )
    
    # Glucose with color based on ROC
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=smooth_data(df, 'glucose_mg_dl', smoothing),
            mode='lines',
            name='Glucose',
            line=dict(color='#6366f1', width=2),
            hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.0f}</b> mg/dL<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Rate of change with color bands
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['roc_smooth'],
            mode='lines',
            name='Rate of Change',
            line=dict(color='#8b5cf6', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.1)',
            hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:+.1f}</b> mg/dL/hr<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add zero line and danger zones
    fig.add_hline(y=0, line_dash="solid", line_color="#d1d5db", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#ef4444", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-30, line_dash="dot", line_color="#ef4444", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12),
        hoverlabel=dict(font_size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    
    return fig


def create_lag_correlation_chart(lag_data: dict) -> go.Figure:
    """Create lag correlation visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lag_data['lags'],
        y=lag_data['correlations'],
        mode='lines+markers',
        name='Correlation',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.1)',
        hovertemplate='Lag: %{x:.1f}h<br>Correlation: %{y:.3f}<extra></extra>'
    ))
    
    # Mark optimal lag
    fig.add_vline(x=lag_data['optimal_lag'], line_dash="dash", line_color="#10b981")
    fig.add_annotation(
        x=lag_data['optimal_lag'],
        y=lag_data['max_correlation'],
        text=f"Optimal: {lag_data['optimal_lag']:.1f}h",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#10b981",
        font=dict(color="#10b981", size=11)
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="#d1d5db")
    
    fig.update_layout(
        title=dict(text="Glucose-Ketone Lag Correlation", font=dict(size=14)),
        xaxis_title="Time Lag (hours, positive = ketones lag behind glucose)",
        yaxis_title="Correlation",
        height=350,
        margin=dict(l=50, r=30, t=60, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bordercolor="rgba(128,128,128,0.3)")
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)', range=[-1, 1])
    
    return fig


def create_rolling_variability_chart(glucose_df: pd.DataFrame, window_hours: int = 24) -> go.Figure:
    """Create rolling variability chart."""
    var_data = calculate_rolling_variability(glucose_df, 'glucose_mg_dl', window_hours)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{window_hours}h Rolling Mean", f"{window_hours}h Rolling CV (%)")
    )
    
    # Rolling mean with std band
    fig.add_trace(
        go.Scatter(
            x=var_data['timestamp'],
            y=var_data['rolling_mean'] + var_data['rolling_std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=var_data['timestamp'],
            y=var_data['rolling_mean'] - var_data['rolling_std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.15)',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=var_data['timestamp'],
            y=var_data['rolling_mean'],
            mode='lines',
            name='Mean',
            line=dict(color='#6366f1', width=2),
            hovertemplate='%{x|%b %d}<br>Mean: <b>%{y:.1f}</b> mg/dL<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Rolling CV
    fig.add_trace(
        go.Scatter(
            x=var_data['timestamp'],
            y=var_data['rolling_cv'],
            mode='lines',
            name='CV',
            line=dict(color='#f97316', width=2),
            fill='tozeroy',
            fillcolor='rgba(249, 115, 22, 0.1)',
            hovertemplate='%{x|%b %d}<br>CV: <b>%{y:.1f}%</b><extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add target lines
    fig.add_hline(y=36, line_dash="dot", line_color="#10b981", opacity=0.7, row=2, col=1)
    fig.add_annotation(x=var_data['timestamp'].iloc[-1], y=36, text="Target <36%", 
                      showarrow=False, font=dict(size=10, color="#10b981"), row=2, col=1)
    
    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    
    return fig


def create_trends_chart(trends_data: dict) -> go.Figure:
    """Create weekly trends chart with regression lines."""
    has_glucose = trends_data.get('glucose') is not None
    has_ketones = trends_data.get('ketones') is not None
    
    if not has_glucose and not has_ketones:
        return None
    
    fig = make_subplots(
        rows=2 if has_glucose else 1, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            (["Daily Mean Glucose", "Daily Mean Ketones"] if has_ketones else ["Daily Mean Glucose"]) 
            if has_glucose else ["Daily Mean Ketones"]
        )
    )
    
    row = 1
    if has_glucose:
        g = trends_data['glucose']
        fig.add_trace(
            go.Bar(
                x=g['dates'],
                y=g['means'],
                name='Glucose',
                marker_color='#6366f1',
                opacity=0.7,
                hovertemplate='%{x}<br>Mean: <b>%{y:.1f}</b> mg/dL<extra></extra>'
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=g['dates'],
                y=g['mean_trend_line'],
                mode='lines',
                name='Trend',
                line=dict(color='#1e1b4b', width=2, dash='dash'),
                hovertemplate='Trend: %{y:.1f}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Add trend annotation
        trend_dir = "↓ Improving" if g['mean_trend_slope'] < 0 else "↑ Rising"
        trend_color = "#10b981" if g['mean_trend_slope'] < 0 else "#f59e0b"
        fig.add_annotation(
            x=g['dates'][-1], y=g['means'][-1],
            text=f"{trend_dir} ({g['mean_trend_slope']:+.1f}/day)",
            showarrow=False,
            font=dict(size=11, color=trend_color),
            xanchor='left',
            row=row, col=1
        )
        row += 1
    
    if has_ketones:
        k = trends_data['ketones']
        fig.add_trace(
            go.Bar(
                x=k['dates'],
                y=k['means'],
                name='Ketones',
                marker_color='#f97316',
                opacity=0.7,
                hovertemplate='%{x}<br>Mean: <b>%{y:.2f}</b> mmol/L<extra></extra>'
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=k['dates'],
                y=k['trend_line'],
                mode='lines',
                name='Trend',
                line=dict(color='#7c2d12', width=2, dash='dash'),
                hovertemplate='Trend: %{y:.2f}<extra></extra>'
            ),
            row=row, col=1
        )
        
        trend_dir = "↑ Improving" if k['trend_slope'] > 0 else "↓ Falling"
        trend_color = "#10b981" if k['trend_slope'] > 0 else "#f59e0b"
        fig.add_annotation(
            x=k['dates'][-1], y=k['means'][-1],
            text=f"{trend_dir} ({k['trend_slope']:+.3f}/day)",
            showarrow=False,
            font=dict(size=11, color=trend_color),
            xanchor='left',
            row=row, col=1
        )
    
    fig.update_layout(
        height=400 if not has_glucose or not has_ketones else 500,
        showlegend=False,
        margin=dict(l=50, r=80, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        bargap=0.3
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
    
    return fig


def create_time_series_plot(glucose_df: pd.DataFrame = None, 
                            ketone_df: pd.DataFrame = None,
                            show_ranges: bool = True,
                            smoothing: int = 1,
                            show_raw: bool = False) -> go.Figure:
    """Create combined time series plot with optional smoothing."""
    fig = make_subplots(
        rows=2 if ketone_df is not None and glucose_df is not None else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            ["Glucose (mg/dL)", "Ketones (mmol/L)"] 
            if ketone_df is not None and glucose_df is not None 
            else ["Glucose (mg/dL)"] if glucose_df is not None 
            else ["Ketones (mmol/L)"]
        )
    )
    
    row = 1
    
    if glucose_df is not None:
        # Show raw data as faint background if requested
        if show_raw and smoothing > 1:
            fig.add_trace(
                go.Scatter(
                    x=glucose_df['timestamp'],
                    y=glucose_df['glucose_mg_dl'],
                    mode='lines',
                    name='Raw',
                    line=dict(color='#6366f1', width=0.5),
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=1
            )
        
        # Smoothed glucose trace
        smoothed_glucose = smooth_data(glucose_df, 'glucose_mg_dl', smoothing)
        fig.add_trace(
            go.Scatter(
                x=glucose_df['timestamp'],
                y=smoothed_glucose,
                mode='lines',
                name='Glucose',
                line=dict(color='#6366f1', width=2, shape='spline', smoothing=0.8),
                hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.0f} mg/dL</b><extra></extra>'
            ),
            row=row, col=1
        )
        
        if show_ranges:
            fig.add_hrect(y0=70, y1=110, fillcolor="rgba(16, 185, 129, 0.06)", 
                         line_width=0, row=row, col=1, 
                         annotation_text="optimal", annotation_position="top left",
                         annotation=dict(font_size=10, font_color="#10b981"))
            fig.add_hline(y=70, line_dash="dot", line_color="#f59e0b", 
                         opacity=0.5, row=row, col=1)
            fig.add_hline(y=140, line_dash="dot", line_color="#f59e0b",
                         opacity=0.5, row=row, col=1)
        
        row += 1
    
    if ketone_df is not None:
        # Show raw data as faint background if requested
        if show_raw and smoothing > 1:
            fig.add_trace(
                go.Scatter(
                    x=ketone_df['timestamp'],
                    y=ketone_df['ketone_mmol_l'],
                    mode='lines',
                    name='Raw',
                    line=dict(color='#f97316', width=0.5),
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=1
            )
        
        # Smoothed ketone trace
        smoothed_ketones = smooth_data(ketone_df, 'ketone_mmol_l', smoothing)
        fig.add_trace(
            go.Scatter(
                x=ketone_df['timestamp'],
                y=smoothed_ketones,
                mode='lines',
                name='Ketones',
                line=dict(color='#f97316', width=2, shape='spline', smoothing=0.8),
                hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.2f} mmol/L</b><extra></extra>'
            ),
            row=row, col=1
        )
        
        if show_ranges:
            fig.add_hrect(y0=0.5, y1=3.0, fillcolor="rgba(249, 115, 22, 0.06)",
                         line_width=0, row=row, col=1)
            fig.add_hline(y=0.5, line_dash="dot", line_color="#10b981",
                         opacity=0.5, row=row, col=1)
            fig.add_hline(y=1.0, line_dash="dot", line_color="#10b981",
                         opacity=0.5, row=row, col=1)
    
    fig.update_layout(
        height=400 if (glucose_df is None or ketone_df is None) else 600,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=50, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12),
        hoverlabel=dict(
            font_size=13,
            font_family="Inter, sans-serif"
        )
    )
    
    # Refined grid
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.15)',
        tickformat='%b %d',
        tickfont=dict(size=11)
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.15)',
        tickfont=dict(size=11)
    )
    
    return fig


def create_overlay_chart(glucose_df: pd.DataFrame = None, 
                         ketone_df: pd.DataFrame = None,
                         show_ranges: bool = True,
                         smoothing: int = 1,
                         show_raw: bool = False) -> go.Figure:
    """Create dual-axis overlay chart with glucose and ketones on same timeline."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if glucose_df is not None:
        # Show raw data as faint background if requested
        if show_raw and smoothing > 1:
            fig.add_trace(
                go.Scatter(
                    x=glucose_df['timestamp'],
                    y=glucose_df['glucose_mg_dl'],
                    mode='lines',
                    name='Raw Glucose',
                    line=dict(color='#6366f1', width=0.5),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
        
        # Smoothed glucose trace
        smoothed_glucose = smooth_data(glucose_df, 'glucose_mg_dl', smoothing)
        fig.add_trace(
            go.Scatter(
                x=glucose_df['timestamp'],
                y=smoothed_glucose,
                mode='lines',
                name='Glucose',
                line=dict(color='#6366f1', width=2.5, shape='spline', smoothing=0.8),
                hovertemplate='<b>%{y:.0f}</b> mg/dL<extra></extra>'
            ),
            secondary_y=False
        )
        
        if show_ranges:
            fig.add_hrect(y0=70, y1=110, fillcolor="rgba(16, 185, 129, 0.08)",
                         line_width=0, secondary_y=False)
            fig.add_hline(y=70, line_dash="dot", line_color="#d1d5db",
                         opacity=0.8, secondary_y=False)
            fig.add_hline(y=140, line_dash="dot", line_color="#d1d5db",
                         opacity=0.8, secondary_y=False)
    
    if ketone_df is not None:
        # Show raw data as faint background if requested
        if show_raw and smoothing > 1:
            fig.add_trace(
                go.Scatter(
                    x=ketone_df['timestamp'],
                    y=ketone_df['ketone_mmol_l'],
                    mode='lines',
                    name='Raw Ketones',
                    line=dict(color='#f97316', width=0.5),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                secondary_y=True
            )
        
        # Smoothed ketone trace
        smoothed_ketones = smooth_data(ketone_df, 'ketone_mmol_l', smoothing)
        fig.add_trace(
            go.Scatter(
                x=ketone_df['timestamp'],
                y=smoothed_ketones,
                mode='lines',
                name='Ketones',
                line=dict(color='#f97316', width=2.5, shape='spline', smoothing=0.8),
                hovertemplate='<b>%{y:.2f}</b> mmol/L<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Set y-axes titles with refined styling
    fig.update_yaxes(
        title_text="Glucose (mg/dL)", 
        secondary_y=False, 
        range=[40, 200] if glucose_df is not None else None,
        color='#6366f1',
        tickfont=dict(size=11),
        title_font=dict(size=12),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.15)'
    )
    fig.update_yaxes(
        title_text="Ketones (mmol/L)", 
        secondary_y=True,
        range=[0, 4] if ketone_df is not None else None,
        color='#f97316',
        tickfont=dict(size=11),
        title_font=dict(size=12),
        showgrid=False
    )
    
    # Add range slider for zooming
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=3, label="3d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            font=dict(size=11),
            borderwidth=1
        ),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.15)',
        tickformat='%b %d, %H:%M',
        tickfont=dict(size=10)
    )
    
    fig.update_layout(
        height=520,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            font=dict(size=12),
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant'
        ),
        hovermode='x unified',
        margin=dict(l=55, r=55, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(
            font_size=13,
            font_family="Inter, sans-serif"
        )
    )
    
    return fig


def create_daily_chart(daily_metrics: list) -> go.Figure:
    """Create daily metrics bar chart."""
    dates = [dm.date for dm in daily_metrics]
    glucose_means = [dm.glucose_mean for dm in daily_metrics]
    ketone_means = [dm.ketone_mean for dm in daily_metrics]
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Daily Mean Glucose", "Daily Mean Ketones")
    )
    
    # Glucose bars
    colors_glucose = [get_glucose_color(v) if v else '#e2e8f0' for v in glucose_means]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=glucose_means,
            name='Glucose',
            marker_color=colors_glucose,
            marker_line=dict(width=0),
            hovertemplate='%{x}<br>Mean: %{y:.1f} mg/dL<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Ketone bars
    colors_ketone = [get_ketone_color(v) if v else '#e2e8f0' for v in ketone_means]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=ketone_means,
            name='Ketones',
            marker_color=colors_ketone,
            marker_line=dict(width=0),
            hovertemplate='%{x}<br>Mean: %{y:.2f} mmol/L<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        bargap=0.15
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    
    return fig


def create_hourly_heatmap(glucose_df: pd.DataFrame = None,
                          ketone_df: pd.DataFrame = None) -> go.Figure:
    """Create hourly pattern heatmap."""
    fig = make_subplots(
        rows=1, 
        cols=2 if glucose_df is not None and ketone_df is not None else 1,
        subplot_titles=(
            ["Glucose by Hour", "Ketones by Hour"]
            if glucose_df is not None and ketone_df is not None
            else ["Glucose by Hour"] if glucose_df is not None
            else ["Ketones by Hour"]
        )
    )
    
    col = 1
    
    if glucose_df is not None:
        df = glucose_df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        
        pivot = df.pivot_table(
            values='glucose_mg_dl',
            index='day',
            columns='hour',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=[[0, '#10b981'], [0.5, '#fef3c7'], [1, '#ef4444']],
                zmin=70,
                zmax=140,
                colorbar=dict(title="mg/dL", x=0.45 if ketone_df is not None else 1.0, tickfont=dict(size=10)),
                hovertemplate='Hour: %{x}<br>Day: %{y}<br>Glucose: %{z:.1f}<extra></extra>'
            ),
            row=1, col=col
        )
        col += 1
    
    if ketone_df is not None:
        df = ketone_df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        
        pivot = df.pivot_table(
            values='ketone_mmol_l',
            index='day',
            columns='hour',
            aggfunc='mean'
        )
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=[[0, '#f8fafc'], [0.5, '#fb923c'], [1, '#ea580c']],
                zmin=0,
                zmax=1.5,
                colorbar=dict(title="mmol/L", tickfont=dict(size=10)),
                hovertemplate='Hour: %{x}<br>Day: %{y}<br>Ketones: %{z:.2f}<extra></extra>'
            ),
            row=1, col=col
        )
    
    fig.update_layout(
        height=350,
        margin=dict(l=100, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    fig.update_xaxes(title_text="Hour of Day")
    
    return fig


def create_distribution_plot(glucose_df: pd.DataFrame = None,
                             ketone_df: pd.DataFrame = None) -> go.Figure:
    """Create distribution histograms."""
    fig = make_subplots(
        rows=1,
        cols=2 if glucose_df is not None and ketone_df is not None else 1,
        subplot_titles=(
            ["Glucose Distribution", "Ketone Distribution"]
            if glucose_df is not None and ketone_df is not None
            else ["Glucose Distribution"] if glucose_df is not None
            else ["Ketone Distribution"]
        )
    )
    
    col = 1
    
    if glucose_df is not None:
        fig.add_trace(
            go.Histogram(
                x=glucose_df['glucose_mg_dl'],
                nbinsx=50,
                name='Glucose',
                marker_color='#6366f1',
                opacity=0.75,
                hovertemplate='Glucose: %{x:.0f} mg/dL<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=col
        )
        
        # Add vertical lines for ranges
        fig.add_vline(x=70, line_dash="dash", line_color="#f59e0b", row=1, col=col)
        fig.add_vline(x=180, line_dash="dash", line_color="#f59e0b", row=1, col=col)
        fig.add_vline(x=110, line_dash="dot", line_color="#10b981", row=1, col=col)
        
        col += 1
    
    if ketone_df is not None:
        fig.add_trace(
            go.Histogram(
                x=ketone_df['ketone_mmol_l'],
                nbinsx=50,
                name='Ketones',
                marker_color='#f97316',
                opacity=0.75,
                hovertemplate='Ketones: %{x:.2f} mmol/L<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=col
        )
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="#10b981", row=1, col=col)
        fig.add_vline(x=1.0, line_dash="dot", line_color="#10b981", row=1, col=col)
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    
    return fig


def create_tir_donut(metrics, metric_type: str = 'glucose') -> go.Figure:
    """Create time-in-range donut chart."""
    if metric_type == 'glucose':
        labels = ['Very Low (<54)', 'Low (54-69)', 'In Range (70-180)', 
                  'High (181-250)', 'Very High (>250)']
        values = [
            metrics.time_very_low,
            metrics.time_low,
            metrics.time_in_range,
            metrics.time_high,
            metrics.time_very_high
        ]
        colors = ['#ef4444', '#f59e0b', '#10b981', '#f97316', '#ef4444']
    else:
        labels = ['Absent (<0.2)', 'Trace (0.2-0.5)', 'Light (0.5-1.0)',
                  'Moderate (1-3)', 'Deep (>3)']
        values = [
            metrics.time_absent,
            metrics.time_trace,
            metrics.time_light,
            metrics.time_moderate,
            metrics.time_deep
        ]
        colors = ['#94a3b8', '#06b6d4', '#10b981', '#22c55e', '#f59e0b']
    
    # Filter out zero values
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0.5]
    if filtered:
        labels, values, colors = zip(*filtered)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker_colors=colors,
        textinfo='percent',
        textposition='outside',
        textfont=dict(size=12, family="Inter, sans-serif"),
        hovertemplate='%{label}<br>%{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.25,
            font=dict(size=11, family="Inter, sans-serif")
        ),
        margin=dict(l=20, r=20, t=20, b=70),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_scatter_glucose_ketone(glucose_df: pd.DataFrame, 
                                   ketone_df: pd.DataFrame) -> go.Figure:
    """Create glucose vs ketone scatter plot for overlapping periods."""
    # Merge on nearest timestamp
    merged = pd.merge_asof(
        glucose_df.sort_values('timestamp'),
        ketone_df.sort_values('timestamp'),
        on='timestamp',
        tolerance=pd.Timedelta(minutes=5),
        direction='nearest'
    ).dropna()
    
    if len(merged) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged['glucose_mg_dl'],
        y=merged['ketone_mmol_l'],
        mode='markers',
        marker=dict(
            size=6,
            color=merged['timestamp'].astype(np.int64),
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title="Time", tickfont=dict(size=10))
        ),
        hovertemplate='Glucose: %{x:.0f} mg/dL<br>Ketones: %{y:.2f} mmol/L<extra></extra>'
    ))
    
    # Add quadrant lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    fig.add_vline(x=100, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    
    # Add quadrant labels
    fig.add_annotation(x=80, y=1.5, text="Optimal<br>Flexibility", 
                      showarrow=False, font=dict(color="#10b981", size=11, family="Inter, sans-serif"))
    fig.add_annotation(x=130, y=0.2, text="Fed<br>State",
                      showarrow=False, font=dict(color="#94a3b8", size=11, family="Inter, sans-serif"))
    
    fig.update_layout(
        title=dict(text="Glucose vs Ketones", font=dict(size=16, family="Inter, sans-serif")),
        xaxis_title="Glucose (mg/dL)",
        yaxis_title="Ketones (mmol/L)",
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    
    return fig


# =============================================================================
# Main App
# =============================================================================

def main():
    st.title("Metabolic Analysis Dashboard")
    st.markdown("Analyze continuous glucose and ketone monitor data")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Data Upload")
        
        dexcom_file = st.file_uploader(
            "Dexcom CGM Export (CSV)",
            type=['csv'],
            help="Upload your Dexcom Clarity CSV export"
        )
        
        sibio_file = st.file_uploader(
            "Sibio Ketone Export (CSV)",
            type=['csv'],
            help="Upload your Sibio ketone monitor CSV export"
        )
        
        st.divider()
        
        st.header("Settings")
        show_ranges = st.checkbox("Show target ranges", value=True)
        
        st.markdown("**Smoothing**")
        smoothing_window = st.slider(
            "Smoothing level", 
            min_value=1, 
            max_value=30, 
            value=5,
            help="Rolling average window. 1 = raw data, higher = smoother curves"
        )
        show_raw_data = st.checkbox("Show raw data behind", value=False)
        
        st.divider()
        
        # Analysis targets
        st.header("Targets")
        glucose_target_low = st.number_input("Glucose low (mg/dL)", value=70, min_value=50, max_value=100)
        glucose_target_high = st.number_input("Glucose high (mg/dL)", value=110, min_value=100, max_value=200)
        ketone_target = st.number_input("Ketone target (mmol/L)", value=0.5, min_value=0.0, max_value=3.0, step=0.1)
        
        st.divider()
        
        # Date range filter
        st.header("Date Range")
        use_date_filter = st.checkbox("Filter by date range", value=False)
        date_range_start = st.date_input("Start date", value=datetime.now() - timedelta(days=7))
        date_range_end = st.date_input("End date", value=datetime.now())
    
    # Load data
    glucose_df = None
    ketone_df = None
    
    if dexcom_file is not None:
        try:
            glucose_df = load_dexcom_data(io.StringIO(dexcom_file.getvalue().decode('utf-8-sig')))
            st.sidebar.success(f"Loaded {len(glucose_df):,} glucose readings")
        except Exception as e:
            st.sidebar.error(f"Error loading Dexcom data: {e}")
    
    if sibio_file is not None:
        try:
            ketone_df = load_sibio_data(io.StringIO(sibio_file.getvalue().decode('utf-8')))
            st.sidebar.success(f"Loaded {len(ketone_df):,} ketone readings")
        except Exception as e:
            st.sidebar.error(f"Error loading Sibio data: {e}")
    
    # Apply date filtering if enabled
    if use_date_filter:
        start_dt = pd.Timestamp(date_range_start)
        end_dt = pd.Timestamp(date_range_end) + pd.Timedelta(days=1)  # Include end date
        
        if glucose_df is not None:
            original_count = len(glucose_df)
            glucose_df = glucose_df[(glucose_df['timestamp'] >= start_dt) & 
                                    (glucose_df['timestamp'] < end_dt)].copy()
            if len(glucose_df) == 0:
                st.sidebar.warning("No glucose data in selected date range")
                glucose_df = None
            else:
                st.sidebar.info(f"Filtered to {len(glucose_df):,} glucose readings")
        
        if ketone_df is not None:
            original_count = len(ketone_df)
            ketone_df = ketone_df[(ketone_df['timestamp'] >= start_dt) & 
                                  (ketone_df['timestamp'] < end_dt)].copy()
            if len(ketone_df) == 0:
                st.sidebar.warning("No ketone data in selected date range")
                ketone_df = None
            else:
                st.sidebar.info(f"Filtered to {len(ketone_df):,} ketone readings")
    
    # Main content
    if glucose_df is None and ketone_df is None:
        st.info("Upload your CGM and/or ketone data files in the sidebar to get started.")
        
        # Show example
        with st.expander("About this tool"):
            st.markdown("""
            This dashboard analyzes continuous glucose monitor (CGM) and continuous ketone monitor data
            to provide insights into metabolic health.
            
            **Supported formats:**
            - **Dexcom**: CSV export from Dexcom Clarity or the Dexcom app
            - **Sibio**: CSV export from Sibio continuous ketone monitor
            
            **Metrics calculated:**
            - Standard CGM metrics (mean, SD, CV, GMI)
            - Time in range analysis
            - Glycemic variability indices (MAGE, J-Index, LBGI/HBGI)
            - Ketone zone distribution
            - Combined metabolic state analysis
            """)
        return
    
    # Run analysis
    glucose_metrics = None
    ketone_metrics = None
    
    if glucose_df is not None:
        glucose_metrics = analyze_glucose(glucose_df)
    
    if ketone_df is not None:
        ketone_metrics = analyze_ketones(ketone_df)
    
    # Calculate daily metrics
    daily_metrics = calculate_daily_metrics(glucose_df, ketone_df)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Advanced Analysis", "Detailed Metrics", "Daily Analysis", 
        "Patterns", "Report"
    ])
    
    # ==========================================================================
    # Tab 1: Overview
    # ==========================================================================
    with tab1:
        # Key metrics row
        if glucose_metrics:
            st.subheader("Glucose Overview")
            st.markdown("""
            **What to look for:** Mean glucose ideally 80-120 mg/dL. CV (coefficient of variation) under 36% indicates stable glucose. 
            GMI estimates your A1C from CGM data. Time in Range (70-180) should be >70% for good control.
            """)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Mean Glucose", f"{glucose_metrics.mean:.1f} mg/dL")
            with col2:
                cv_delta = "Good" if glucose_metrics.cv < 36 else "High"
                st.metric("CV", f"{glucose_metrics.cv:.1f}%", delta=cv_delta, 
                         delta_color="off" if glucose_metrics.cv < 36 else "inverse")
            with col3:
                st.metric("GMI (est. A1C)", f"{glucose_metrics.gmi:.2f}%")
            with col4:
                st.metric("Time in Range", f"{glucose_metrics.time_in_range:.1f}%")
            with col5:
                custom_tir = (np.sum((glucose_df['glucose_mg_dl'] >= glucose_target_low) & 
                                     (glucose_df['glucose_mg_dl'] <= glucose_target_high)) / 
                             len(glucose_df)) * 100
                st.metric(f"Time {glucose_target_low}-{glucose_target_high}", f"{custom_tir:.1f}%")
        
        if ketone_metrics:
            st.subheader("Ketone Overview")
            st.markdown("""
            **What to look for:** On a ketogenic diet, aim for 0.5-3.0 mmol/L (nutritional ketosis). 
            Higher isn't always better — consistency matters more than peaks. Mean ketones of 0.5-1.5 mmol/L is typical for well-adapted individuals.
            """)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Mean Ketones", f"{ketone_metrics.mean:.2f} mmol/L")
            with col2:
                st.metric("Median", f"{ketone_metrics.median:.2f} mmol/L")
            with col3:
                st.metric("Peak", f"{ketone_metrics.peak_ketone:.2f} mmol/L")
            with col4:
                st.metric(f"Time ≥{ketone_target}", f"{ketone_metrics.time_above_1:.1f}%" 
                         if ketone_target == 1.0 else 
                         f"{(np.sum(ketone_df['ketone_mmol_l'] >= ketone_target) / len(ketone_df) * 100):.1f}%")
            with col5:
                st.metric("Time in Ketosis", f"{ketone_metrics.time_light + ketone_metrics.time_moderate + ketone_metrics.time_deep:.1f}%")
        
        # Overlay chart (glucose + ketones on same timeline with zoom)
        if glucose_df is not None and ketone_df is not None:
            st.subheader("Glucose & Ketones Overlay")
            st.markdown("""
            **What to look for:** When glucose drops, ketones should rise (and vice versa) — this shows your body switching fuel sources. 
            Look for the inverse relationship: low glucose periods should align with elevated ketones. 
            Use the time buttons (6h, 1d, 3d, 1w) or drag the slider below to zoom into specific meals or fasting periods.
            """)
            overlay_fig = create_overlay_chart(glucose_df, ketone_df, show_ranges, smoothing_window, show_raw_data)
            st.plotly_chart(overlay_fig, use_container_width=True)
        
        # Time series plot (separate panels)
        st.subheader("Time Series (Separate Panels)")
        st.markdown("""
        **What to look for:** The shaded zones show target ranges. For glucose, green (70-180) is the standard target zone. 
        Red areas indicate time spent too high or too low. Smoother lines with fewer sharp spikes indicate better metabolic control.
        """)
        fig = create_time_series_plot(glucose_df, ketone_df, show_ranges, smoothing_window, show_raw_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            if glucose_metrics:
                st.subheader("Glucose Time in Range")
                st.markdown("**Takeaway:** The green slice shows time in the healthy 70-180 mg/dL range. Aim for >70% green.")
                fig = create_tir_donut(glucose_metrics, 'glucose')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if ketone_metrics:
                st.subheader("Ketone Zones")
                st.markdown("**Takeaway:** Light, moderate, and deep ketosis zones show fat-burning states. Trace/absent means glucose-burning mode.")
                fig = create_tir_donut(ketone_metrics, 'ketone')
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # Tab 2: Advanced Analysis
    # ==========================================================================
    with tab2:
        st.subheader("Advanced Metabolic Analysis")
        st.markdown("""
        These analyses dig deeper into your metabolic patterns to help you understand how your body responds to food, fasting, and daily rhythms.
        """)
        
        # Metabolic Flexibility Score
        overlap_data = analyze_overlap_periods(glucose_df, ketone_df) if glucose_df is not None and ketone_df is not None else None
        flex_score = calculate_metabolic_flexibility_score(glucose_metrics, ketone_metrics, overlap_data)
        
        if flex_score['total'] is not None:
            st.markdown("### Metabolic Flexibility Score")
            st.markdown("""
            **What this means:** This score measures how well your body can switch between burning glucose and fat for fuel. 
            A higher score indicates better metabolic health. 
            - **Glucose Stability** (40 pts): Low variability and time in range
            - **Ketone Production** (30 pts): Ability to produce ketones consistently  
            - **Flexibility** (30 pts): The inverse relationship between glucose and ketones (when one goes down, the other goes up)
            """)
            
            score_cols = st.columns(4)
            with score_cols[0]:
                st.metric(
                    "Overall Score", 
                    f"{flex_score['total']:.0f} / {flex_score['max_possible']}",
                    help="Composite score combining glucose stability, ketone production, and metabolic flexibility"
                )
            with score_cols[1]:
                if flex_score['glucose_stability'] is not None:
                    st.metric("Glucose Stability", f"{flex_score['glucose_stability']:.0f} / 40")
            with score_cols[2]:
                if flex_score['ketone_production'] is not None:
                    st.metric("Ketone Production", f"{flex_score['ketone_production']:.0f} / 30")
            with score_cols[3]:
                if flex_score['flexibility'] is not None:
                    st.metric("Flexibility", f"{flex_score['flexibility']:.0f} / 30")
            
            st.divider()
        
        # Rate of Change Analysis
        if glucose_df is not None:
            st.markdown("### Rate of Change")
            st.markdown("""
            **What this shows:** How fast your glucose rises and falls, measured in mg/dL per hour. 
            - **Green bars** = glucose rising (after eating)
            - **Red bars** = glucose falling (insulin response or fasting)
            - The dotted lines at ±30 mg/dL/hr mark rapid changes that may indicate problematic spikes or crashes
            
            **Takeaway:** Smaller, more gradual changes indicate better glycemic control. Large swings suggest meals that spike your blood sugar.
            """)
            
            roc_fig = create_rate_of_change_chart(glucose_df, smoothing_window)
            st.plotly_chart(roc_fig, use_container_width=True)
            
            st.divider()
        
        # Lag Correlation (requires both glucose and ketones)
        if glucose_df is not None and ketone_df is not None:
            st.markdown("### Glucose-Ketone Lag Analysis")
            st.markdown("""
            **What this shows:** This measures how your ketone levels correlate with glucose at different time offsets.
            A negative correlation means when glucose is high, ketones are low (and vice versa) — this is the healthy, expected pattern.
            
            **Takeaway:** The "optimal lag" tells you how long it takes your body to shift into ketone production after glucose drops. 
            Typically 2-4 hours. A stronger negative correlation (closer to -1) indicates better metabolic flexibility.
            """)
            
            lag_data = calculate_lag_correlation(glucose_df, ketone_df)
            
            if lag_data['correlations']:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    lag_fig = create_lag_correlation_chart(lag_data)
                    st.plotly_chart(lag_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Interpretation**")
                    st.metric("Optimal Lag", f"{lag_data['optimal_lag']:.1f} hours")
                    st.metric("Peak Correlation", f"{lag_data['max_correlation']:.3f}")
                    
                    if lag_data['optimal_lag'] > 0:
                        st.info(f"Ketones respond to glucose changes with a ~{abs(lag_data['optimal_lag']):.1f}h delay")
                    else:
                        st.info("Ketones and glucose are synchronously correlated")
            
            st.divider()
        
        # Rolling Variability
        if glucose_df is not None:
            st.markdown("### Rolling Variability")
            st.markdown("""
            **What this shows:** The top chart shows your 24-hour rolling average glucose. The bottom shows your coefficient of variation (CV) over time.
            
            **Takeaway:** CV under 36% is the clinical target for good glucose control (shown by the green dotted line). 
            If your CV is trending downward over time, your glucose stability is improving. 
            Periods of high CV often correspond to dietary changes, stress, or poor sleep.
            """)
            
            var_fig = create_rolling_variability_chart(glucose_df, 24)
            st.plotly_chart(var_fig, use_container_width=True)
            
            st.divider()
        
        # Weekly Trends
        trends = calculate_weekly_trends(glucose_df, ketone_df)
        if trends['glucose'] or trends['ketones']:
            st.markdown("### Weekly Trends")
            st.markdown("""
            **What this shows:** Each bar represents your daily average. The trend line shows whether you're improving over time.
            
            **Takeaway:** For glucose, a downward-sloping trend line means your average blood sugar is decreasing (good!). 
            For ketones, an upward trend indicates you're spending more time in ketosis. 
            Focus on the direction of the trend, not day-to-day fluctuations.
            """)
            
            trends_fig = create_trends_chart(trends)
            if trends_fig:
                st.plotly_chart(trends_fig, use_container_width=True)
            
            # Trend summary
            trend_cols = st.columns(2)
            with trend_cols[0]:
                if trends['glucose']:
                    g = trends['glucose']
                    status = "Improving" if g['improving'] else "Needs attention"
                    color = "normal" if g['improving'] else "off"
                    st.metric("Glucose Trend", status, f"{g['mean_trend_slope']:+.1f} mg/dL per day", delta_color=color)
            with trend_cols[1]:
                if trends['ketones']:
                    k = trends['ketones']
                    status = "Improving" if k['improving'] else "Declining"
                    color = "normal" if k['improving'] else "off"
                    st.metric("Ketone Trend", status, f"{k['trend_slope']:+.3f} mmol/L per day", delta_color=color)
            
            st.divider()
        
        # Glucose Spike Analysis
        if glucose_df is not None:
            st.markdown("### Glucose Spike Analysis")
            st.markdown("""
            **What this shows:** Detects when your glucose rose rapidly (≥30 mg/dL), how high it went, and how long it took to recover.
            
            **Takeaway:** Frequent spikes suggest foods or meals that your body struggles to handle. 
            Note the times — are spikes happening after specific meals? 
            Fast recovery times (<60 min) indicate good insulin sensitivity. Slow recovery (>90 min) may suggest insulin resistance.
            """)
            spikes = analyze_glucose_spikes(glucose_df)
            
            if spikes:
                st.caption(f"Detected {len(spikes)} significant glucose spikes (rise ≥30 mg/dL)")
                
                # Summary metrics
                avg_rise = np.mean([s['rise'] for s in spikes])
                avg_ttp = np.mean([s['time_to_peak_min'] for s in spikes])
                recovery_times = [s['recovery_time_min'] for s in spikes if s['recovery_time_min'] is not None]
                avg_recovery = np.mean(recovery_times) if recovery_times else None
                
                spike_cols = st.columns(4)
                with spike_cols[0]:
                    st.metric("Spikes Detected", len(spikes))
                with spike_cols[1]:
                    st.metric("Avg Rise", f"{avg_rise:.0f} mg/dL")
                with spike_cols[2]:
                    st.metric("Avg Time to Peak", f"{avg_ttp:.0f} min")
                with spike_cols[3]:
                    if avg_recovery:
                        st.metric("Avg Recovery", f"{avg_recovery:.0f} min")
                
                # Spike table
                with st.expander("View all spikes"):
                    spike_df = pd.DataFrame(spikes)
                    spike_df['start_time'] = spike_df['start_time'].dt.strftime('%b %d, %H:%M')
                    spike_df['peak_time'] = spike_df['peak_time'].dt.strftime('%H:%M')
                    st.dataframe(spike_df, hide_index=True, use_container_width=True)
            else:
                st.success("No significant glucose spikes detected!")
            
            st.divider()
        
        # Overnight Analysis
        overnight = analyze_overnight(glucose_df, ketone_df)
        if overnight:
            st.markdown("### Overnight Analysis (12am - 6am)")
            st.markdown("""
            **What this shows:** Your glucose patterns during sleep, when you're not eating. This reveals your baseline metabolic state.
            
            **Takeaway:** 
            - **Mean** should be 70-100 mg/dL during sleep
            - **CV** under 36% indicates stable overnight glucose
            - **Dawn Effect** is the natural rise in glucose before waking (liver releasing glucose). >10 mg/dL rise may indicate insulin resistance.
            - **Night Ketones** tend to be highest overnight since you're fasting while sleeping
            """)
            
            night_cols = st.columns(5)
            with night_cols[0]:
                st.metric("Mean", f"{overnight['overnight_mean']:.0f} mg/dL")
            with night_cols[1]:
                st.metric("CV", f"{overnight['overnight_cv']:.1f}%")
            with night_cols[2]:
                st.metric("Range", f"{overnight['overnight_min']:.0f} - {overnight['overnight_max']:.0f}")
            with night_cols[3]:
                if 'dawn_phenomenon' in overnight:
                    dawn = overnight['dawn_phenomenon']
                    st.metric("Dawn Effect", f"{dawn:+.0f} mg/dL", 
                             delta="Normal" if abs(dawn) < 10 else "Elevated",
                             delta_color="off" if abs(dawn) >= 10 else "normal")
            with night_cols[4]:
                if 'overnight_ketone_mean' in overnight:
                    st.metric("Night Ketones", f"{overnight['overnight_ketone_mean']:.2f} mmol/L")
            
            st.divider()
        
        # Fasting Windows
        st.markdown("### Detected Fasting Windows")
        st.markdown("""
        **What this shows:** Periods where your body was clearly in a fasted, fat-burning state — low glucose AND elevated ketones simultaneously.
        
        **Takeaway:** These windows show when you were truly fasting (not just not eating, but metabolically fasted). 
        More total fasting hours generally means more time burning fat for fuel. 
        If you're doing intermittent fasting, check if your fasting windows are actually producing ketones.
        """)
        fasting_windows = detect_fasting_windows(glucose_df, ketone_df)
        
        if fasting_windows:
            st.caption(f"Found {len(fasting_windows)} fasting periods (≥4 hours with glucose <95 mg/dL and ketones ≥0.5 mmol/L)")
            
            fasting_df = pd.DataFrame(fasting_windows)
            fasting_df['start'] = fasting_df['start'].dt.strftime('%b %d, %H:%M')
            fasting_df['end'] = fasting_df['end'].dt.strftime('%b %d, %H:%M')
            fasting_df.columns = ['Start', 'End', 'Duration (h)', 'Avg Glucose', 'Min Glucose', 'Avg Ketones']
            st.dataframe(fasting_df, hide_index=True, use_container_width=True)
            
            # Summary
            total_fasting = sum([w['duration_hours'] for w in fasting_windows])
            st.info(f"Total time in detected fasting state: **{total_fasting:.1f} hours**")
        else:
            st.info("No extended fasting windows detected. Fasting requires glucose <95 mg/dL AND ketones ≥0.5 mmol/L for ≥4 hours.")
    
    # ==========================================================================
    # Tab 3: Detailed Metrics
    # ==========================================================================
    with tab3:
        st.markdown("""
        **Reference Guide:** This tab provides the complete statistical breakdown of your data. 
        Use it to track specific numbers over time or share with your healthcare provider.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if glucose_metrics:
                st.subheader("Glucose Metrics")
                st.markdown("""
                **Key targets:**
                - Mean: 80-120 mg/dL (non-diabetic), <154 mg/dL (diabetic)
                - CV: <36% (stable), <33% (excellent)
                - GMI: <7% (good), <6.5% (excellent)
                """)
                
                st.markdown("**Basic Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'CV', 'GMI'],
                    'Value': [
                        f"{glucose_metrics.mean:.1f} mg/dL",
                        f"{glucose_metrics.median:.1f} mg/dL",
                        f"{glucose_metrics.std:.1f} mg/dL",
                        f"{glucose_metrics.cv:.1f}%",
                        f"{glucose_metrics.gmi:.2f}%"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("**Time in Range**")
                tir_df = pd.DataFrame({
                    'Range': ['Very Low (<54)', 'Low (54-69)', 'In Range (70-180)', 
                              'High (181-250)', 'Very High (>250)', 
                              'Tight (70-140)', 'Optimal (70-110)'],
                    'Percentage': [
                        f"{glucose_metrics.time_very_low:.1f}%",
                        f"{glucose_metrics.time_low:.1f}%",
                        f"{glucose_metrics.time_in_range:.1f}%",
                        f"{glucose_metrics.time_high:.1f}%",
                        f"{glucose_metrics.time_very_high:.1f}%",
                        f"{glucose_metrics.time_tight_range:.1f}%",
                        f"{glucose_metrics.time_optimal:.1f}%"
                    ]
                })
                st.dataframe(tir_df, hide_index=True, use_container_width=True)
                
                st.markdown("**Variability Indices**")
                var_df = pd.DataFrame({
                    'Index': ['MAGE', 'J-Index', 'LBGI', 'HBGI'],
                    'Value': [
                        f"{glucose_metrics.mage:.1f} mg/dL",
                        f"{glucose_metrics.j_index:.1f}",
                        f"{glucose_metrics.lbgi:.2f}",
                        f"{glucose_metrics.hbgi:.2f}"
                    ],
                    'Interpretation': [
                        'Good' if glucose_metrics.mage < 30 else 'Elevated',
                        'Good' if glucose_metrics.j_index < 20 else 'Elevated',
                        'Low risk' if glucose_metrics.lbgi < 2.5 else 'Elevated risk',
                        'Low risk' if glucose_metrics.hbgi < 4.5 else 'Elevated risk'
                    ]
                })
                st.dataframe(var_df, hide_index=True, use_container_width=True)
                
                st.markdown("**Data Quality**")
                st.write(f"Readings: {glucose_metrics.readings_count:,}")
                st.write(f"Coverage: {glucose_metrics.data_coverage:.1f}%")
                st.write(f"Period: {glucose_metrics.date_range[0].strftime('%Y-%m-%d')} to {glucose_metrics.date_range[1].strftime('%Y-%m-%d')}")
        
        with col2:
            if ketone_metrics:
                st.subheader("Ketone Metrics")
                
                st.markdown("**Basic Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{ketone_metrics.mean:.2f} mmol/L",
                        f"{ketone_metrics.median:.2f} mmol/L",
                        f"{ketone_metrics.std:.2f} mmol/L",
                        f"{ketone_metrics.min_val:.2f} mmol/L",
                        f"{ketone_metrics.max_val:.2f} mmol/L"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("**Time in Ketone Zones**")
                zones_df = pd.DataFrame({
                    'Zone': ['Absent (<0.2)', 'Trace (0.2-0.5)', 'Light (0.5-1.0)',
                            'Moderate (1.0-3.0)', 'Deep (>3.0)'],
                    'Percentage': [
                        f"{ketone_metrics.time_absent:.1f}%",
                        f"{ketone_metrics.time_trace:.1f}%",
                        f"{ketone_metrics.time_light:.1f}%",
                        f"{ketone_metrics.time_moderate:.1f}%",
                        f"{ketone_metrics.time_deep:.1f}%"
                    ]
                })
                st.dataframe(zones_df, hide_index=True, use_container_width=True)
                
                st.markdown("**Therapeutic Thresholds**")
                thresh_df = pd.DataFrame({
                    'Threshold': ['≥0.5 mmol/L', '≥1.0 mmol/L', '≥2.0 mmol/L'],
                    'Time': [
                        f"{ketone_metrics.time_light + ketone_metrics.time_moderate + ketone_metrics.time_deep:.1f}%",
                        f"{ketone_metrics.time_above_1:.1f}%",
                        f"{ketone_metrics.time_above_2:.1f}%"
                    ]
                })
                st.dataframe(thresh_df, hide_index=True, use_container_width=True)
                
                st.markdown("**Peak Analysis**")
                st.write(f"Peak: {ketone_metrics.peak_ketone:.2f} mmol/L")
                st.write(f"Time: {ketone_metrics.peak_timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                st.markdown("**Data Quality**")
                st.write(f"Readings: {ketone_metrics.readings_count:,}")
                st.write(f"Coverage: {ketone_metrics.data_coverage:.1f}%")
                st.write(f"Period: {ketone_metrics.date_range[0].strftime('%Y-%m-%d')} to {ketone_metrics.date_range[1].strftime('%Y-%m-%d')}")
        
        # Combined analysis
        if glucose_df is not None and ketone_df is not None:
            st.subheader("Combined Metabolic Analysis")
            
            overlap = analyze_overlap_periods(glucose_df, ketone_df)
            
            if overlap.get('overlap_readings', 0) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overlapping Readings", f"{overlap['overlap_readings']:,}")
                    st.metric("Overlap Period", f"{overlap['overlap_hours']:.1f} hours")
                
                with col2:
                    st.metric("Glucose-Ketone Correlation", f"{overlap['glucose_ketone_correlation']:.3f}")
                    if overlap.get('mean_glucose_when_ketones_elevated'):
                        st.metric("Mean Glucose (ketones ≥0.5)", 
                                 f"{overlap['mean_glucose_when_ketones_elevated']:.1f} mg/dL")
                
                with col3:
                    if overlap.get('mean_ketones_when_glucose_low'):
                        st.metric("Mean Ketones (glucose <100)", 
                                 f"{overlap['mean_ketones_when_glucose_low']:.2f} mmol/L")
                
                st.markdown("**Metabolic State Distribution**")
                st.markdown("*This shows what percentage of time you spent in each metabolic state based on glucose and ketone levels.*")
                states = overlap.get('metabolic_states_pct', {})
                states_df = pd.DataFrame({
                    'State': [k.replace('_', ' ').title() for k in states.keys()],
                    'Percentage': [f"{v:.1f}%" for v in states.values()]
                })
                st.dataframe(states_df, hide_index=True, use_container_width=True)
                
                # Scatter plot
                st.markdown("**Glucose vs Ketones Scatter Plot**")
                st.markdown("""
                *Each dot is a moment in time. The bottom-right quadrant (low glucose, high ketones) represents the ideal fasted/ketogenic state. 
                Top-left (high glucose, low ketones) is the fed/glucose-burning state.*
                """)
                fig = create_scatter_glucose_ketone(glucose_df, ketone_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No overlapping time periods found between glucose and ketone data.")
        
        # Distributions
        st.subheader("Distributions")
        st.markdown("*Histograms showing how often each glucose and ketone level occurs. A tighter, narrower distribution indicates more stable values.*")
        fig = create_distribution_plot(glucose_df, ketone_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # Tab 4: Daily Analysis
    # ==========================================================================
    with tab4:
        st.subheader("Daily Breakdown")
        st.markdown("""
        **What this shows:** Your glucose and ketone averages broken down by day. 
        
        **How to use it:** Look for patterns — are weekends different from weekdays? 
        Which days had the best control? Use this to identify what you did differently on good days vs. bad days.
        """)
        
        if daily_metrics:
            # Daily chart
            fig = create_daily_chart(daily_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily table
            st.subheader("Daily Data Table")
            st.markdown("*Tip: Look for days with low glucose mean AND high ketone mean — those are your best metabolic days.*")
            
            daily_data = []
            for dm in daily_metrics:
                row = {'Date': dm.date}
                if dm.glucose_mean is not None:
                    row['Glucose Mean'] = f"{dm.glucose_mean:.1f}"
                    row['Glucose SD'] = f"{dm.glucose_std:.1f}" if dm.glucose_std else "-"
                    row['Glucose Min'] = f"{dm.glucose_min:.0f}" if dm.glucose_min else "-"
                    row['Glucose Max'] = f"{dm.glucose_max:.0f}" if dm.glucose_max else "-"
                    row['TIR (%)'] = f"{dm.glucose_tir:.1f}" if dm.glucose_tir else "-"
                    row['G Readings'] = dm.glucose_readings
                if dm.ketone_mean is not None:
                    row['Ketone Mean'] = f"{dm.ketone_mean:.2f}"
                    row['Ketone Max'] = f"{dm.ketone_max:.2f}" if dm.ketone_max else "-"
                    row['K Readings'] = dm.ketone_readings
                daily_data.append(row)
            
            daily_df = pd.DataFrame(daily_data)
            st.dataframe(daily_df, hide_index=True, use_container_width=True)
            
            # Download button
            csv = daily_df.to_csv(index=False)
            st.download_button(
                "Download Daily Data (CSV)",
                csv,
                "daily_metrics.csv",
                "text/csv"
            )
    
    # ==========================================================================
    # Tab 5: Patterns
    # ==========================================================================
    with tab5:
        st.subheader("Hourly Patterns")
        st.markdown("""
        **What this shows:** Your average glucose and ketone levels at each hour of the day, aggregated across all days in your data.
        
        **How to use it:** 
        - **Heatmap:** Darker colors = higher values. Look for consistent patterns (e.g., always high after lunch).
        - **Peak/Nadir hours:** When is your glucose highest and lowest? This reveals your body's daily rhythm.
        - **Dawn phenomenon:** The natural rise in glucose before waking. >10 mg/dL rise is considered elevated.
        
        **Takeaway:** Use this to time your meals, fasting windows, and activities. If glucose peaks at 1pm, consider what you're eating for lunch.
        """)
        
        # Heatmap
        fig = create_hourly_heatmap(glucose_df, ketone_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly analysis
        hourly = analyze_hourly_patterns(glucose_df, ketone_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'glucose_by_hour' in hourly:
                st.markdown("**Glucose Patterns**")
                st.write(f"Peak hour: {hourly['glucose_peak_hour']}:00")
                st.write(f"Nadir hour: {hourly['glucose_nadir_hour']}:00")
                st.write(f"Dawn phenomenon: {hourly['dawn_phenomenon']:+.1f} mg/dL")
                
                # Hourly line chart
                hourly_glucose = pd.DataFrame(hourly['glucose_by_hour'])
                fig = px.line(hourly_glucose, x=hourly_glucose.index, y='mean',
                             title='Average Glucose by Hour',
                             labels={'index': 'Hour', 'mean': 'Glucose (mg/dL)'})
                fig.add_scatter(x=hourly_glucose.index, 
                               y=hourly_glucose['mean'] + hourly_glucose['std'],
                               mode='lines', line=dict(width=0), showlegend=False)
                fig.add_scatter(x=hourly_glucose.index,
                               y=hourly_glucose['mean'] - hourly_glucose['std'],
                               mode='lines', line=dict(width=0), fill='tonexty',
                               fillcolor='rgba(31, 119, 180, 0.2)', showlegend=False)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'ketone_by_hour' in hourly:
                st.markdown("**Ketone Patterns**")
                st.write(f"Peak hour: {hourly['ketone_peak_hour']}:00")
                st.write(f"Nadir hour: {hourly['ketone_nadir_hour']}:00")
                
                hourly_ketone = pd.DataFrame(hourly['ketone_by_hour'])
                fig = px.line(hourly_ketone, x=hourly_ketone.index, y='mean',
                             title='Average Ketones by Hour',
                             labels={'index': 'Hour', 'mean': 'Ketones (mmol/L)'})
                fig.add_scatter(x=hourly_ketone.index,
                               y=hourly_ketone['mean'] + hourly_ketone['std'],
                               mode='lines', line=dict(width=0), showlegend=False)
                fig.add_scatter(x=hourly_ketone.index,
                               y=hourly_ketone['mean'] - hourly_ketone['std'],
                               mode='lines', line=dict(width=0), fill='tonexty',
                               fillcolor='rgba(255, 127, 14, 0.2)', showlegend=False)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # Tab 6: Report
    # ==========================================================================
    with tab6:
        st.subheader("Full Report")
        st.markdown("""
        **What this is:** A comprehensive text summary of all your metrics, ready to download and share with your healthcare provider or keep for your records.
        
        **Formats available:**
        - **TXT:** Human-readable report with all key findings
        - **JSON:** Machine-readable data for importing into other tools or tracking over time
        """)
        
        # Generate analysis
        analysis = type('Analysis', (), {
            'glucose_metrics': glucose_metrics,
            'ketone_metrics': ketone_metrics,
            'daily_metrics': daily_metrics,
            'overlap_analysis': analyze_overlap_periods(glucose_df, ketone_df) if glucose_df is not None and ketone_df is not None else None
        })()
        
        report = generate_report(analysis)
        
        st.code(report, language=None)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "Download Report (TXT)",
                report,
                "metabolic_report.txt",
                "text/plain"
            )
        
        with col2:
            # JSON export
            import json
            json_data = {
                'glucose_metrics': glucose_metrics.to_dict() if glucose_metrics else None,
                'ketone_metrics': ketone_metrics.to_dict() if ketone_metrics else None,
                'overlap_analysis': analysis.overlap_analysis
            }
            st.download_button(
                "Download Data (JSON)",
                json.dumps(json_data, indent=2),
                "metabolic_data.json",
                "application/json"
            )


if __name__ == "__main__":
    main()
