# CGM/CKM Analyzer

A modular Python package and Streamlit dashboard for analyzing continuous glucose monitor (CGM) and ketone monitor data.

## Features

- **Evidence-Based Metrics**: Organized by evidence tier (Consensus → Optimization → Experimental)
- **Dual Visualization**: Interactive Plotly charts + publication-quality Matplotlib plots
- **Modular Architecture**: Reusable analyzers, loaders, and visualizers

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run cgm_ckm_analyzer/app.py

# Or use the helper script
python run_dashboard.py
```

Upload your Dexcom CGM and/or Sibio ketone CSV files in the sidebar.

## Supported Data Sources

- **Dexcom G7/G6**: CSV export from Dexcom Clarity
- **Sibio**: CSV export from Sibio continuous ketone monitor

## Project Structure

```
cgm_ckm_analyzer/
├── app.py                 # Streamlit dashboard
├── config.py              # Configuration dataclasses
├── config.yaml            # Default thresholds (editable)
├── analyzers/
│   ├── glucose.py         # CGM metrics (GMI, TIR, CV, MAGE, etc.)
│   ├── ketone.py          # Ketone zone analysis
│   └── combined.py        # Joint glucose+ketone analysis
├── metrics/
│   ├── glucose_metrics.py # Glucose metrics dataclass
│   ├── ketone_metrics.py  # Ketone metrics dataclass
│   └── daily_metrics.py   # Daily aggregation dataclass
├── loaders/
│   ├── dexcom.py          # Dexcom CSV parser
│   └── sibio.py           # Sibio CSV parser
├── visualizers/
│   ├── plotly_viz.py      # Interactive charts
│   └── matplotlib_viz.py  # Publication-quality plots
├── utils/
│   ├── smoothing.py       # Savitzky-Golay, rolling average
│   ├── statistics.py      # CV, quantiles, AUC calculations
│   └── colors.py          # Color palettes, evidence badges
└── reports/
    └── generator.py       # Text report generation
```

## Evidence Tiers

Metrics are organized by strength of clinical evidence:

| Tier | Description | Examples |
|------|-------------|----------|
| 🟢 **Consensus** | ADA/EASD/International Guidelines | TIR, GMI, CV, LBGI/HBGI |
| 🟡 **Optimization** | Metabolic health literature | Tight range (70-140), MAGE, ketone zones |
| 🔴 **Experimental** | Novel/unvalidated analyses | Metabolic Flexibility Score, lag correlation |

## Python API

```python
from cgm_ckm_analyzer.config import AnalysisConfig, load_config
from cgm_ckm_analyzer.analyzers import GlucoseAnalyzer, KetoneAnalyzer
from cgm_ckm_analyzer.visualizers import MatplotlibVisualizer
import pandas as pd

# Load your data
glucose_df = pd.read_csv('dexcom_export.csv')
# ... preprocess to have 'timestamp' and 'glucose_mg_dl' columns

# Analyze
config = load_config()  # or AnalysisConfig() for defaults
analyzer = GlucoseAnalyzer(glucose_df, config)
metrics = analyzer.metrics

print(f"GMI: {metrics.gmi:.1f}%")
print(f"TIR: {metrics.time_in_range:.1f}%")
print(f"CV: {metrics.cv:.1f}%")

# Visualize
viz = MatplotlibVisualizer(config)
fig = viz.create_daily_overlay(glucose_df)
fig.savefig('cgm_overlay.png', dpi=300)
```

## Configuration

Edit `cgm_ckm_analyzer/config.yaml` to customize thresholds:

```yaml
glucose:
  target_high: 180      # Standard TIR upper (consensus)
  tight_high: 140       # Tighter target (optimization)
  optimal_high: 110     # Longevity target (optimization)

ketones:
  light_ketosis: 0.5    # Nutritional ketosis threshold
  therapeutic: 1.0      # Therapeutic ketosis threshold
```

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- scipy
- matplotlib
- plotly
- pyyaml

## License

MIT
