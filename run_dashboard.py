#!/usr/bin/env python
"""
Run the Metabolic Analysis Dashboard

Usage:
    streamlit run run_dashboard.py
    
    or
    
    python run_dashboard.py  # Will invoke streamlit
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Path to the app
    app_path = Path(__file__).parent / "metabolic_dashboard" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--browser.gatherUsageStats", "false"
    ])


if __name__ == "__main__":
    main()
