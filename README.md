# Metabolic Analysis Module

Python module and Streamlit dashboard for analyzing continuous glucose monitor (CGM) and continuous ketone monitor (CKM) data.

## Quick Start - Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then upload your Dexcom and/or Sibio CSV files in the sidebar.

![Dashboard Screenshot](screenshot.png)

## Supported Data Sources

- **Dexcom G7**: CSV export from Dexcom Clarity or app
- **Sibio**: CSV export from Sibio continuous ketone monitor

## Installation

```bash
# Requires pandas and numpy
pip install pandas numpy
```

## Usage

### Command Line

```bash
# Analyze both glucose and ketone data
python metabolic_analysis.py --dexcom dexcom_export.csv --sibio sibio_export.csv

# JSON output
python metabolic_analysis.py --dexcom dexcom_export.csv --sibio sibio_export.csv --output json

# Save to file
python metabolic_analysis.py --dexcom dexcom_export.csv --output text --save report.txt

# Analyze only glucose
python metabolic_analysis.py --dexcom dexcom_export.csv

# Analyze only ketones
python metabolic_analysis.py --sibio sibio_export.csv
```

### Python API

```python
from metabolic_analysis import (
    run_analysis,
    generate_report,
    load_dexcom_data,
    load_sibio_data,
    analyze_glucose,
    analyze_ketones,
)

# Full analysis
analysis = run_analysis(
    dexcom_path='dexcom_export.csv',
    sibio_path='sibio_export.csv'
)

# Access results
print(analysis.glucose_metrics.mean)        # 97.3
print(analysis.glucose_metrics.gmi)         # 5.64
print(analysis.ketone_metrics.time_above_1) # 7.5

# Generate text report
report = generate_report(analysis)
print(report)

# Get as dict for JSON serialization
glucose_dict = analysis.glucose_metrics.to_dict()
ketone_dict = analysis.ketone_metrics.to_dict()
```

## Metrics Computed

### Glucose Metrics

| Metric | Description |
|--------|-------------|
| Mean, Median, SD | Basic descriptive statistics |
| CV (%) | Coefficient of variation - key variability metric |
| GMI | Glucose Management Indicator (estimated A1C) |
| Time in Range | Standard ranges: <54, 54-69, 70-180, 181-250, >250 |
| Tight Range | 70-140 mg/dL (metabolic optimization target) |
| Optimal | 70-110 mg/dL (longevity optimization target) |
| MAGE | Mean Amplitude of Glycemic Excursions |
| J-Index | Combined mean + variability metric |
| LBGI/HBGI | Low/High Blood Glucose Risk Index |

### Ketone Metrics

| Metric | Description |
|--------|-------------|
| Mean, Median, SD, Range | Basic descriptive statistics |
| Time in Zones | Absent (<0.2), Trace (0.2-0.5), Light (0.5-1.0), Moderate (1-3), Deep (>3) |
| Peak | Maximum ketone reading with timestamp |
| Time ≥1.0 | Therapeutic ketosis threshold |
| Time ≥2.0 | Deep ketosis threshold |

### Combined Analysis (when both data sources available)

- Metabolic state classification during overlapping periods
- Glucose-ketone correlation
- Mean glucose when ketones elevated
- Mean ketones when glucose low

## Metabolic State Classifications

| State | Definition |
|-------|------------|
| Optimal Flexibility | Glucose <100 + Ketones >0.5 |
| Fasted/Keto | Glucose <100 + Ketones 0.5-3.0 |
| Fed State | Glucose >120 + Ketones <0.3 |
| Stress Response | Glucose >140 + Ketones >1.0 |
| Moderate Mixed | All other combinations |

## Data Format Requirements

### Dexcom CSV

Standard Clarity export format with columns:
- `Timestamp (YYYY-MM-DDThh:mm:ss)`
- `Event Type` (filter for "EGV")
- `Glucose Value (mg/dL)`

### Sibio CSV

Standard export with columns:
- `Time` (YYYY-MM-DD HH:MM:SS)
- `Sensor reading(mmol/L)`

## Example Output

```
======================================================================
METABOLIC ANALYSIS REPORT
======================================================================

GLUCOSE METRICS (Dexcom CGM)
----------------------------------------
Date Range: 2025-12-16 to 2025-12-29
Readings: 3,592 (91.0% coverage)

Basic Statistics:
  Mean Glucose:     97.3 mg/dL
  Median Glucose:   97.0 mg/dL
  Std Deviation:    11.3 mg/dL
  CV:               11.6%
  GMI (est. A1C):   5.64%

Time in Range:
  Very Low (<54):   0.0%
  Low (54-69):      1.1%
  In Range (70-180): 98.9%
  High (181-250):   0.0%
  Very High (>250): 0.0%

Optimization Targets:
  Tight Range (70-140): 98.7%
  Optimal (70-110):     88.5%
...
```

## Extending the Module

The module is designed to be extensible. Key extension points:

1. **New data sources**: Add loader functions following the pattern of `load_dexcom_data()` and `load_sibio_data()`
2. **Additional metrics**: Add calculation functions and extend the dataclasses
3. **Visualization**: The data structures are designed to work easily with matplotlib/plotly

## References

- International Consensus on CGM metrics (Battelino et al., 2019)
- GMI formula (Bergenstal et al., 2018)
- BGRI calculations (Kovatchev et al.)
