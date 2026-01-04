"""
Matplotlib Visualizer - Publication-quality CGM plots.

Preserves the aesthetic from cgm_plot.py:
- Savitzky-Golay smoothing
- Time-of-day gradient backgrounds (night/sunrise/day/sunset)
- Quantile ribbons
- Avenir font family
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from io import BytesIO

from cgm_ckm_analyzer.config import AnalysisConfig
from cgm_ckm_analyzer.utils.smoothing import savgol_smooth, smooth_by_time_of_day
from cgm_ckm_analyzer.utils.statistics import remove_outliers_iqr_grouped
from cgm_ckm_analyzer.utils.colors import (
    NIGHT_COLOR, DAYTIME_COLOR, SUNRISE_COLOR, SUNSET_COLOR,
    CGM_LINE_COLOR, CGM_MAX_LINE_COLOR,
    interpolate_colors, get_time_of_day_periods, get_glucose_reference_lines,
)


class MatplotlibVisualizer:
    """Publication-quality matplotlib visualizations.
    
    Produces the original cgm_plot.py aesthetic with:
    - Savitzky-Golay smoothed curves
    - Time-of-day gradient backgrounds
    - Quantile ribbons (10th, 34th, 50th, 68th, 90th percentiles)
    - Reference lines at key glucose thresholds
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize visualizer.
        
        Args:
            config: Optional configuration.
        """
        self.config = config or AnalysisConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style."""
        try:
            mpl.rcParams['font.family'] = self.config.visualization.font_family
        except:
            mpl.rcParams['font.family'] = 'sans-serif'
    
    def create_daily_overlay(
        self,
        df: pd.DataFrame,
        ketone_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        show_quantiles: bool = True,
        show_max_line: bool = True,
        show_reference_lines: bool = True,
        show_gradient_background: bool = True,
        figsize: Tuple[int, int] = (15, 8),
    ) -> plt.Figure:
        """Create daily overlay plot showing glucose patterns by time of day.
        
        This is the signature visualization from cgm_plot.py, showing:
        - Median (50th percentile) as main line
        - Shaded quantile bands
        - Optional max values line
        - Time-of-day gradient background
        - Reference lines at key thresholds
        - Optional smoothed ketone curve on secondary y-axis
        
        Args:
            df: DataFrame with 'timestamp' and 'glucose_mg_dl' columns.
            ketone_df: Optional DataFrame with 'timestamp' and 'ketone_mmol_l' columns.
            title: Optional plot title.
            show_quantiles: Show quantile ribbons.
            show_max_line: Show smoothed maximum line.
            show_reference_lines: Show horizontal reference lines.
            show_gradient_background: Show day/night gradient.
            figsize: Figure size.
        
        Returns:
            Matplotlib Figure object.
        """
        viz = self.config.visualization
        
        # Prepare data
        df = df.copy()
        df['minute_of_day'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        
        # Remove outliers per minute group
        df = remove_outliers_iqr_grouped(
            df, 'glucose_mg_dl', 'minute_of_day', factor=viz.iqr_factor
        )
        
        # Calculate quantiles by minute
        grouped = df.groupby('minute_of_day')['glucose_mg_dl']
        quantiles_list = list(viz.quantiles)
        quantiles = grouped.quantile(quantiles_list).unstack()
        
        # Apply Savitzky-Golay smoothing
        window = viz.savgol_window
        polyorder = viz.savgol_polyorder
        
        # Ensure window is valid
        if len(quantiles) < window:
            window = len(quantiles) if len(quantiles) % 2 == 1 else len(quantiles) - 1
        if window <= polyorder:
            window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        
        smoothed_quantiles = quantiles.copy()
        for col in smoothed_quantiles.columns:
            try:
                from scipy.signal import savgol_filter
                smoothed_quantiles[col] = savgol_filter(
                    quantiles[col].interpolate().values,
                    window,
                    polyorder
                )
            except:
                pass  # Keep unsmoothed if filter fails
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Add gradient background
        if show_gradient_background:
            self._add_time_gradient(ax)
        
        # Plot median line
        ax.plot(
            smoothed_quantiles.index,
            smoothed_quantiles[0.5],
            label='Median',
            color=CGM_LINE_COLOR,
            alpha=0.75,
            linewidth=2
        )
        
        # Plot quantile ribbons
        if show_quantiles and len(quantiles_list) >= 4:
            # Shade area above 100 mg/dL (from 68th percentile)
            ax.fill_between(
                smoothed_quantiles.index,
                np.where(smoothed_quantiles[0.68] > 100, smoothed_quantiles[0.68], 100),
                100,
                where=smoothed_quantiles[0.68] > 100,
                color='grey',
                alpha=0.45,
                label='Area above 100 mg/dL (+1 SD)'
            )
        
        # Plot smoothed max line
        if show_max_line:
            max_values = df.groupby('minute_of_day')['glucose_mg_dl'].max()
            try:
                from scipy.signal import savgol_filter
                smoothed_max = savgol_filter(
                    max_values.interpolate().values,
                    window,
                    polyorder
                )
                ax.plot(
                    max_values.index,
                    smoothed_max,
                    label='Smoothed Maximum',
                    color=CGM_MAX_LINE_COLOR,
                    alpha=0.25,
                    linewidth=2,
                    linestyle='--'
                )
            except:
                pass
        
        # Add reference lines
        if show_reference_lines:
            for value, color, linestyle, alpha in get_glucose_reference_lines():
                ax.axhline(y=value, color=color, linestyle=linestyle, alpha=alpha)
        
        # Format x-axis as time
        ax.set_xticks([i for i in range(0, 24*60+1, 60)])
        ax.set_xticklabels([f'{i:02}:00' for i in range(25)], rotation=45)
        ax.set_xlim(smoothed_quantiles.index.min(), smoothed_quantiles.index.max())
        
        # Labels
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Glucose (mg/dL)")
        
        # Add ketone data on secondary y-axis if provided
        ax2 = None
        if ketone_df is not None and len(ketone_df) > 0:
            ax2 = ax.twinx()
            
            # Prepare ketone data
            ketone_df = ketone_df.copy()
            ketone_df['minute_of_day'] = ketone_df['timestamp'].dt.hour * 60 + ketone_df['timestamp'].dt.minute
            
            # Calculate median by minute
            ketone_grouped = ketone_df.groupby('minute_of_day')['ketone_mmol_l']
            ketone_median = ketone_grouped.median()
            
            # Smooth the ketone curve
            if len(ketone_median) >= 5:
                try:
                    from scipy.signal import savgol_filter
                    k_window = min(51, len(ketone_median) if len(ketone_median) % 2 == 1 else len(ketone_median) - 1)
                    k_polyorder = min(3, k_window - 1)
                    smoothed_ketones = savgol_filter(
                        ketone_median.interpolate().values,
                        k_window,
                        k_polyorder
                    )
                    ketone_x = ketone_median.index
                except:
                    smoothed_ketones = ketone_median.values
                    ketone_x = ketone_median.index
            else:
                smoothed_ketones = ketone_median.values
                ketone_x = ketone_median.index
            
            # Plot smoothed ketone curve
            ax2.plot(
                ketone_x,
                smoothed_ketones,
                color='#10b981',  # Emerald green
                linewidth=2.5,
                alpha=0.8,
                label='Ketones (median)'
            )
            
            # Style secondary axis
            ax2.set_ylabel("Ketones (mmol/L)", color='#10b981')
            ax2.tick_params(axis='y', labelcolor='#10b981')
            ax2.set_ylim(0, max(3.0, np.max(smoothed_ketones) * 1.2))
            ax2.spines['right'].set_color('#10b981')
            
            # Add ketone reference line at 0.5 (nutritional ketosis)
            ax2.axhline(y=0.5, color='#10b981', linestyle=':', alpha=0.5)
        
        # Legends
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        if ax2 is not None:
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # Title
        if title:
            ax.set_title(title, weight='bold')
        else:
            date_range = self._get_date_range_str(df)
            ketone_suffix = " + Ketones" if ketone_df is not None else ""
            ax.set_title(
                f"CGM Daily Overlay{ketone_suffix} | {date_range} | Savitzky-Golay Smoothing",
                weight='bold'
            )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return fig
    
    def _add_time_gradient(self, ax: plt.Axes):
        """Add time-of-day gradient background to axes.
        
        Creates sunrise/day/sunset/night gradient effect.
        """
        periods = get_time_of_day_periods()
        
        for start, end, start_color, end_color in periods:
            gradient = interpolate_colors(start_color, end_color, end - start)
            for i, color in enumerate(gradient):
                ax.axvspan(start + i, start + i + 1, facecolor=color, alpha=0.40)
    
    def _get_date_range_str(self, df: pd.DataFrame) -> str:
        """Get formatted date range string."""
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        return f"{min_date.strftime('%m/%d/%Y')} to {max_date.strftime('%m/%d/%Y')}"
    
    def create_time_series(
        self,
        df: pd.DataFrame,
        column: str = 'glucose_mg_dl',
        title: Optional[str] = None,
        show_smoothed: bool = True,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """Create time series plot.
        
        Args:
            df: DataFrame with 'timestamp' and value column.
            column: Column name for y-axis values.
            title: Optional plot title.
            show_smoothed: Overlay smoothed line.
            figsize: Figure size.
        
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot raw data
        ax.plot(
            df['timestamp'],
            df[column],
            alpha=0.4,
            linewidth=1,
            color='#888888',
            label='Raw'
        )
        
        # Plot smoothed
        if show_smoothed:
            smoothed = savgol_smooth(df[column])
            ax.plot(
                df['timestamp'],
                smoothed,
                linewidth=2,
                color=CGM_LINE_COLOR,
                label='Smoothed'
            )
        
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        ax.legend()
        
        if title:
            ax.set_title(title, weight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_histogram(
        self,
        values: np.ndarray,
        bins: int = 50,
        title: str = "Glucose Distribution",
        xlabel: str = "Glucose (mg/dL)",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Create histogram of values.
        
        Args:
            values: Array of values.
            bins: Number of histogram bins.
            title: Plot title.
            xlabel: X-axis label.
            figsize: Figure size.
        
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(values, bins=bins, edgecolor='white', alpha=0.7, color=CGM_LINE_COLOR)
        
        # Add reference lines
        for value, color, _, _ in get_glucose_reference_lines():
            ax.axvline(x=value, color=color, linestyle='--', alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title, weight='bold')
        
        plt.tight_layout()
        return fig
    
    def save_figure(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        dpi: int = 300,
    ):
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure.
            filepath: Output file path.
            dpi: Resolution.
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    def figure_to_bytes(self, fig: plt.Figure, format: str = 'png') -> bytes:
        """Convert figure to bytes for embedding.
        
        Args:
            fig: Matplotlib figure.
            format: Image format ('png', 'svg', etc.).
        
        Returns:
            Image bytes.
        """
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
