"""
Sibio continuous ketone monitor data loader.

Parses Sibio app CSV exports into standardized DataFrame format.
"""

import pandas as pd
from pathlib import Path
from typing import Union


class SibioLoader:
    """Loader for Sibio continuous ketone monitor CSV exports.
    
    Sibio is a continuous ketone monitor that exports data similar to CGM.
    """
    
    # Known column name variations
    TIMESTAMP_COLUMNS = [
        'Time',
        'Timestamp',
        'DateTime',
    ]
    
    KETONE_COLUMNS = [
        'Sensor reading(mmol/L)',
        'Sensor Reading (mmol/L)',
        'Ketone Value',
        'Ketone',
    ]
    
    def __init__(self, filepath: Union[str, Path]):
        """Initialize loader with file path.
        
        Args:
            filepath: Path to Sibio CSV export.
        """
        self.filepath = Path(filepath)
        self._df: pd.DataFrame = None
    
    def load(self) -> pd.DataFrame:
        """Load and parse Sibio CSV.
        
        Returns:
            DataFrame with columns: timestamp, ketone_mmol_l
        """
        # Read CSV - Sibio may have trailing commas creating empty columns
        df = pd.read_csv(self.filepath)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Find the correct column names
        timestamp_col = self._find_column(df, self.TIMESTAMP_COLUMNS)
        ketone_col = self._find_column(df, self.KETONE_COLUMNS)
        
        if timestamp_col is None or ketone_col is None:
            raise ValueError(
                f"Could not find required columns in Sibio file. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Extract ketone value
        df['ketone_mmol_l'] = pd.to_numeric(df[ketone_col], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=['timestamp', 'ketone_mmol_l'])
        
        # Filter out invalid readings (negative or extremely high)
        df = df[(df['ketone_mmol_l'] >= 0) & (df['ketone_mmol_l'] < 20)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self._df = df[['timestamp', 'ketone_mmol_l']]
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
