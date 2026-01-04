"""
Dexcom CGM data loader.

Parses Dexcom Clarity CSV exports into standardized DataFrame format.
"""

import pandas as pd
from pathlib import Path
from typing import Union


class DexcomLoader:
    """Loader for Dexcom CGM CSV exports.
    
    Dexcom Clarity exports include various event types:
    - EGV: Estimated Glucose Values (continuous readings)
    - Calibration: Fingerstick calibrations
    - Insulin: Insulin doses
    - Carbs: Carbohydrate entries
    
    This loader extracts only EGV readings for CGM analysis.
    """
    
    # Known column name variations across Dexcom versions
    TIMESTAMP_COLUMNS = [
        'Timestamp (YYYY-MM-DDThh:mm:ss)',
        'Timestamp',
        'DateTime',
    ]
    
    GLUCOSE_COLUMNS = [
        'Glucose Value (mg/dL)',
        'Glucose Value',
        'EGV',
    ]
    
    EVENT_TYPE_COLUMNS = [
        'Event Type',
        'EventType',
        'Type',
    ]
    
    def __init__(self, filepath: Union[str, Path]):
        """Initialize loader with file path.
        
        Args:
            filepath: Path to Dexcom CSV export.
        """
        self.filepath = Path(filepath)
        self._df: pd.DataFrame = None
    
    def load(self) -> pd.DataFrame:
        """Load and parse Dexcom CSV.
        
        Returns:
            DataFrame with columns: timestamp, glucose_mg_dl
        """
        # Read with UTF-8-BOM handling (Dexcom exports often have BOM)
        df = pd.read_csv(self.filepath, encoding='utf-8-sig')
        
        # Find the correct column names
        timestamp_col = self._find_column(df, self.TIMESTAMP_COLUMNS)
        glucose_col = self._find_column(df, self.GLUCOSE_COLUMNS)
        event_col = self._find_column(df, self.EVENT_TYPE_COLUMNS)
        
        if timestamp_col is None or glucose_col is None:
            raise ValueError(
                f"Could not find required columns in Dexcom file. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Filter for EGV (glucose readings) only if event type column exists
        if event_col is not None:
            df = df[df[event_col] == 'EGV'].copy()
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Extract glucose value
        df['glucose_mg_dl'] = pd.to_numeric(df[glucose_col], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=['timestamp', 'glucose_mg_dl'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle duplicate timestamps (take the mean)
        df = df.groupby('timestamp').agg({
            'glucose_mg_dl': 'mean'
        }).reset_index()
        
        self._df = df[['timestamp', 'glucose_mg_dl']]
        return self._df
    
    def _find_column(self, df: pd.DataFrame, candidates: list) -> str:
        """Find matching column name from candidates.
        
        Args:
            df: DataFrame to search.
            candidates: List of possible column names.
        
        Returns:
            Matching column name or None.
        """
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
    
    @property
    def df(self) -> pd.DataFrame:
        """Get loaded DataFrame (loads on first access)."""
        if self._df is None:
            self._df = self.load()
        return self._df
    
    def get_date_range(self) -> tuple:
        """Get date range of data.
        
        Returns:
            Tuple of (start_date, end_date).
        """
        df = self.df
        return (
            df['timestamp'].min().to_pydatetime(),
            df['timestamp'].max().to_pydatetime(),
        )
    
    def get_readings_count(self) -> int:
        """Get total number of readings."""
        return len(self.df)
