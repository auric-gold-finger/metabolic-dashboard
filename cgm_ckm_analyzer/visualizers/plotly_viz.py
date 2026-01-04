"""
Plotly Visualizer - Interactive charts for Streamlit dashboard.

Provides all the interactive chart functions for the web app.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, Tuple

from cgm_ckm_analyzer.config import AnalysisConfig
from cgm_ckm_analyzer.utils.smoothing import rolling_smooth
from cgm_ckm_analyzer.utils.colors import (
    get_glucose_color,
    get_ketone_color,
    GLUCOSE_COLORSCALE,
    KETONE_COLORSCALE,
    EVIDENCE_TIER_COLORS,
)


class PlotlyVisualizer:
    """Interactive Plotly visualizations for Streamlit.
    
    Provides interactive charts with consistent styling.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize visualizer.
        
        Args:
            config: Optional configuration.
        """
        self.config = config or AnalysisConfig()
        self.font_family = "Inter, sans-serif"
    
    def _get_base_layout(self, height: int = 400, **kwargs) -> dict:
        """Get base layout for consistent styling."""
        return {
            'height': height,
            'margin': dict(l=50, r=30, t=50, b=30),
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': dict(family=self.font_family, size=12),
            'hoverlabel': dict(font_size=12, bordercolor='rgba(128,128,128,0.3)'),
            **kwargs
        }
    
    def _smooth(self, df: pd.DataFrame, column: str, window: int = 5) -> pd.Series:
        """Apply smoothing to a column."""
        if window <= 1:
            return df[column]
        return rolling_smooth(df[column], window=window)
    
    # =========================================================================
    # TIME SERIES CHARTS
    # =========================================================================
    
    def create_overlay_chart(
        self,
        glucose_df: Optional[pd.DataFrame],
        ketone_df: Optional[pd.DataFrame],
        smoothing: int = 5,
        height: int = 500,
    ) -> go.Figure:
        """Create dual-axis overlay chart for glucose and ketones.
        
        Args:
            glucose_df: DataFrame with glucose data.
            ketone_df: DataFrame with ketone data.
            smoothing: Smoothing window size.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if glucose_df is not None and len(glucose_df) > 0:
            smoothed = self._smooth(glucose_df, 'glucose_mg_dl', smoothing)
            
            fig.add_trace(
                go.Scatter(
                    x=glucose_df['timestamp'],
                    y=smoothed,
                    name='Glucose',
                    mode='lines',
                    line=dict(color='#6366f1', width=2),
                    hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.0f}</b> mg/dL<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add glucose reference zones
            self._add_glucose_zones(fig)
        
        if ketone_df is not None and len(ketone_df) > 0:
            smoothed = self._smooth(ketone_df, 'ketone_mmol_l', smoothing)
            
            fig.add_trace(
                go.Scatter(
                    x=ketone_df['timestamp'],
                    y=smoothed,
                    name='Ketones',
                    mode='lines',
                    line=dict(color='#10b981', width=2),
                    hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.2f}</b> mmol/L<extra></extra>'
                ),
                secondary_y=True
            )
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Glucose (mg/dL)", secondary_y=False)
        fig.update_yaxes(title_text="Ketones (mmol/L)", secondary_y=True)
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        
        return fig
    
    def _add_glucose_zones(self, fig: go.Figure):
        """Add glucose reference zones to a figure."""
        t = self.config.glucose
        
        # Add horizontal reference lines
        fig.add_hline(y=t.very_low, line_dash='dash', line_color='#ef4444', opacity=0.5)
        fig.add_hline(y=t.low, line_dash='dash', line_color='#f59e0b', opacity=0.5)
        fig.add_hline(y=t.target_high, line_dash='dash', line_color='#f59e0b', opacity=0.5)
        fig.add_hline(y=t.very_high, line_dash='dash', line_color='#ef4444', opacity=0.5)
    
    def create_time_series_stacked(
        self,
        glucose_df: Optional[pd.DataFrame],
        ketone_df: Optional[pd.DataFrame],
        smoothing: int = 5,
        height: int = 600,
    ) -> go.Figure:
        """Create stacked time series plot with separate panels.
        
        Args:
            glucose_df: DataFrame with glucose data.
            ketone_df: DataFrame with ketone data.
            smoothing: Smoothing window size.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        rows = sum([glucose_df is not None, ketone_df is not None])
        if rows == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=['Glucose (mg/dL)', 'Ketones (mmol/L)'][:rows]
        )
        
        row = 1
        
        if glucose_df is not None and len(glucose_df) > 0:
            smoothed = self._smooth(glucose_df, 'glucose_mg_dl', smoothing)
            
            fig.add_trace(
                go.Scatter(
                    x=glucose_df['timestamp'],
                    y=smoothed,
                    mode='lines',
                    name='Glucose',
                    line=dict(color='#6366f1', width=2),
                    hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.0f}</b> mg/dL<extra></extra>'
                ),
                row=row, col=1
            )
            row += 1
        
        if ketone_df is not None and len(ketone_df) > 0:
            smoothed = self._smooth(ketone_df, 'ketone_mmol_l', smoothing)
            
            fig.add_trace(
                go.Scatter(
                    x=ketone_df['timestamp'],
                    y=smoothed,
                    mode='lines',
                    name='Ketones',
                    line=dict(color='#10b981', width=2),
                    hovertemplate='%{x|%b %d, %H:%M}<br><b>%{y:.2f}</b> mmol/L<extra></extra>'
                ),
                row=row, col=1
            )
        
        fig.update_layout(**self._get_base_layout(height=height), showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        
        return fig
    
    def create_rate_of_change_chart(
        self,
        df: pd.DataFrame,
        column: str = 'glucose_mg_dl',
        smoothing: int = 5,
        height: int = 450,
    ) -> go.Figure:
        """Create rate of change visualization.
        
        Args:
            df: DataFrame with timestamp and value column.
            column: Column name for values.
            smoothing: Smoothing window size.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        df = df.copy()
        
        # Calculate rate of change
        time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
        value_diff = df[column].diff()
        df['roc'] = (value_diff / time_diff).fillna(0)
        
        # Smooth
        df['roc_smooth'] = self._smooth(df, 'roc', smoothing)
        df['value_smooth'] = self._smooth(df, column, smoothing)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=('Values', 'Rate of Change (per hour)')
        )
        
        # Values
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['value_smooth'],
                mode='lines',
                name='Value',
                line=dict(color='#6366f1', width=2),
            ),
            row=1, col=1
        )
        
        # Rate of change
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['roc_smooth'],
                mode='lines',
                name='Rate of Change',
                line=dict(color='#8b5cf6', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)',
            ),
            row=2, col=1
        )
        
        # Reference lines
        fig.add_hline(y=0, line_dash='solid', line_color='#d1d5db', row=2, col=1)
        fig.add_hline(y=30, line_dash='dot', line_color='#ef4444', opacity=0.5, row=2, col=1)
        fig.add_hline(y=-30, line_dash='dot', line_color='#ef4444', opacity=0.5, row=2, col=1)
        
        fig.update_layout(**self._get_base_layout(height=height), showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        
        return fig
    
    def create_rolling_variability_chart(
        self,
        df: pd.DataFrame,
        column: str = 'glucose_mg_dl',
        window_hours: int = 24,
        height: int = 450,
    ) -> go.Figure:
        """Create rolling variability chart.
        
        Args:
            df: DataFrame with data.
            column: Column name.
            window_hours: Rolling window in hours.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        # Estimate readings per hour
        total_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        readings_per_hour = len(df) / total_hours if total_hours > 0 else 12
        window_size = max(1, int(window_hours * readings_per_hour))
        
        # Calculate rolling stats
        rolling = df[column].rolling(window=window_size, center=True, min_periods=window_size // 2)
        
        var_data = pd.DataFrame({
            'timestamp': df['timestamp'],
            'rolling_mean': rolling.mean(),
            'rolling_std': rolling.std(),
            'rolling_cv': (rolling.std() / rolling.mean()) * 100,
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{window_hours}h Rolling Mean', f'{window_hours}h Rolling CV (%)')
        )
        
        # Mean with std band
        fig.add_trace(
            go.Scatter(
                x=var_data['timestamp'],
                y=var_data['rolling_mean'] + var_data['rolling_std'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=var_data['timestamp'],
                y=var_data['rolling_mean'] - var_data['rolling_std'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(99, 102, 241, 0.15)',
                showlegend=False, hoverinfo='skip'
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
            ),
            row=1, col=1
        )
        
        # CV
        fig.add_trace(
            go.Scatter(
                x=var_data['timestamp'],
                y=var_data['rolling_cv'],
                mode='lines',
                name='CV',
                line=dict(color='#f97316', width=2),
                fill='tozeroy',
                fillcolor='rgba(249, 115, 22, 0.1)',
            ),
            row=2, col=1
        )
        
        # CV target line
        fig.add_hline(
            y=self.config.analysis.cv_target,
            line_dash='dash',
            line_color='#10b981',
            opacity=0.7,
            row=2, col=1,
            annotation_text=f"Target: {self.config.analysis.cv_target}%"
        )
        
        fig.update_layout(**self._get_base_layout(height=height), showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        
        return fig
    
    # =========================================================================
    # DISTRIBUTION CHARTS
    # =========================================================================
    
    def create_histogram(
        self,
        values: np.ndarray,
        title: str = 'Distribution',
        xlabel: str = 'Value',
        color: str = '#6366f1',
        height: int = 350,
    ) -> go.Figure:
        """Create histogram.
        
        Args:
            values: Array of values.
            title: Chart title.
            xlabel: X-axis label.
            color: Bar color.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=50,
            marker_color=color,
            opacity=0.75,
            hovertemplate='%{x:.0f}: %{y} readings<extra></extra>'
        ))
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            title=dict(text=title, font=dict(size=14)),
            xaxis_title=xlabel,
            yaxis_title='Frequency',
            bargap=0.05,
        )
        
        return fig
    
    def create_donut_chart(
        self,
        labels: List[str],
        values: List[float],
        colors: List[str],
        title: str = 'Time in Range',
        height: int = 300,
    ) -> go.Figure:
        """Create donut chart for time in range.
        
        Args:
            labels: Zone labels.
            values: Percentages for each zone.
            colors: Colors for each zone.
            title: Chart title.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=colors,
            textinfo='percent',
            textposition='inside',
            hovertemplate='%{label}<br><b>%{value:.1f}%</b><extra></extra>'
        )])
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            title=dict(text=title, font=dict(size=14)),
            showlegend=True,
            legend=dict(orientation='h', y=-0.1),
        )
        
        return fig
    
    def create_glucose_tir_donut(self, metrics) -> go.Figure:
        """Create glucose Time in Range donut chart.
        
        Args:
            metrics: GlucoseMetrics object.
        
        Returns:
            Plotly Figure.
        """
        labels = ['Very Low', 'Low', 'In Range', 'High', 'Very High']
        values = [
            metrics.time_very_low,
            metrics.time_low,
            metrics.time_in_range,
            metrics.time_high,
            metrics.time_very_high,
        ]
        colors = ['#ef4444', '#f59e0b', '#10b981', '#f97316', '#ef4444']
        
        return self.create_donut_chart(labels, values, colors, 'Time in Range')
    
    def create_ketone_zones_donut(self, metrics) -> go.Figure:
        """Create ketone zones donut chart.
        
        Args:
            metrics: KetoneMetrics object.
        
        Returns:
            Plotly Figure.
        """
        labels = ['Absent', 'Trace', 'Light', 'Moderate', 'Deep']
        values = [
            metrics.time_absent,
            metrics.time_trace,
            metrics.time_light,
            metrics.time_moderate,
            metrics.time_deep,
        ]
        colors = ['#94a3b8', '#06b6d4', '#10b981', '#22c55e', '#f59e0b']
        
        return self.create_donut_chart(labels, values, colors, 'Ketone Zones')
    
    # =========================================================================
    # CORRELATION AND ANALYTICS
    # =========================================================================
    
    def create_scatter_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = 'X',
        ylabel: str = 'Y',
        title: str = 'Scatter Plot',
        height: int = 400,
    ) -> go.Figure:
        """Create scatter plot.
        
        Args:
            x: X values.
            y: Y values.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Chart title.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=6,
                color='#6366f1',
                opacity=0.5,
            ),
            hovertemplate=f'{xlabel}: %{{x:.1f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>'
        ))
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            title=dict(text=title, font=dict(size=14)),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        
        return fig
    
    def create_lag_correlation_chart(
        self,
        lag_data: Dict[str, Any],
        height: int = 350,
    ) -> go.Figure:
        """Create lag correlation visualization.
        
        Args:
            lag_data: Dictionary with correlations, lags, optimal_lag, max_correlation.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
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
        if lag_data.get('optimal_lag') is not None:
            fig.add_vline(
                x=lag_data['optimal_lag'],
                line_dash='dash',
                line_color='#10b981'
            )
            fig.add_annotation(
                x=lag_data['optimal_lag'],
                y=lag_data['max_correlation'],
                text=f"Optimal: {lag_data['optimal_lag']:.1f}h",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#10b981',
                font=dict(color='#10b981', size=11)
            )
        
        fig.add_hline(y=0, line_dash='solid', line_color='#d1d5db')
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            title=dict(text='Glucose-Ketone Lag Correlation', font=dict(size=14)),
            xaxis_title='Time Lag (hours)',
            yaxis_title='Correlation',
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)', range=[-1, 1])
        
        return fig
    
    # =========================================================================
    # HEATMAPS
    # =========================================================================
    
    def create_hourly_heatmap(
        self,
        df: pd.DataFrame,
        column: str,
        title: str = 'Hourly Pattern',
        colorscale: str = 'Viridis',
        height: int = 350,
    ) -> go.Figure:
        """Create day×hour heatmap.
        
        Args:
            df: DataFrame with timestamp and value column.
            column: Column name.
            title: Chart title.
            colorscale: Plotly colorscale name.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        df = df.copy()
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        pivot = df.pivot_table(
            values=column,
            index='date',
            columns='hour',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(range(24)),
            y=[str(d) for d in pivot.index],
            colorscale=colorscale,
            hovertemplate='Hour: %{x}<br>Date: %{y}<br>Value: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            title=dict(text=title, font=dict(size=14)),
            xaxis_title='Hour of Day',
            yaxis_title='Date',
        )
        
        return fig
    
    # =========================================================================
    # TRENDS
    # =========================================================================
    
    def create_trends_chart(
        self,
        trends_data: Dict[str, Any],
        height: int = 400,
    ) -> go.Figure:
        """Create weekly trends visualization.
        
        Args:
            trends_data: Dictionary with glucose and/or ketone trend data.
            height: Chart height.
        
        Returns:
            Plotly Figure.
        """
        rows = sum(['glucose' in trends_data, 'ketones' in trends_data])
        if rows == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=[k.title() for k in ['glucose', 'ketones'] if k in trends_data]
        )
        
        row = 1
        
        if 'glucose' in trends_data:
            data = trends_data['glucose']
            
            fig.add_trace(
                go.Scatter(
                    x=data['dates'],
                    y=data['means'],
                    mode='markers+lines',
                    name='Daily Mean',
                    line=dict(color='#6366f1', width=1),
                    marker=dict(size=8),
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['dates'],
                    y=data['trend_line'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='#10b981' if data.get('improving') else '#ef4444', width=2, dash='dash'),
                ),
                row=row, col=1
            )
            row += 1
        
        if 'ketones' in trends_data:
            data = trends_data['ketones']
            
            fig.add_trace(
                go.Scatter(
                    x=data['dates'],
                    y=data['means'],
                    mode='markers+lines',
                    name='Daily Mean',
                    line=dict(color='#10b981', width=1),
                    marker=dict(size=8),
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['dates'],
                    y=data['trend_line'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='#10b981' if data.get('improving') else '#f59e0b', width=2, dash='dash'),
                ),
                row=row, col=1
            )
        
        fig.update_layout(
            **self._get_base_layout(height=height),
            showlegend=False,
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        
        return fig
