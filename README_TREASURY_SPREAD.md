# Treasury Spread Recession Prediction

## Overview

The `treasury_spread_recession_prediction.py` script is a financial analysis tool designed to predict economic recessions using the **Treasury Yield Curve Spread** as an indicator. This script analyzes historical U.S. Treasury yield data to identify recession signals based on the spread between long-term and short-term interest rates.

## Table of Contents

- [What This Script Does](#what-this-script-does)
- [Economic Theory](#economic-theory)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Example Usage](#example-usage)
- [Understanding the Results](#understanding-the-results)
- [Output Files](#output-files)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [References](#references)

## What This Script Does

This script performs the following analyses:

1. **Data Retrieval**: Fetches historical U.S. Treasury yield data (typically 10-year and 2-year yields)
2. **Spread Calculation**: Computes the difference between long-term and short-term Treasury yields
3. **Recession Identification**: Identifies historical recession periods using NBER (National Bureau of Economic Research) data
4. **Signal Detection**: Analyzes when yield curve inversion occurred (negative spreads) as a recession predictor
5. **Statistical Analysis**: Calculates correlation between yield spread and recession occurrence
6. **Visualization**: Generates charts showing historical spread trends and recession periods
7. **Predictive Analysis**: Provides insights into recession probability based on current market conditions

### Key Features

- **Historical Analysis**: Examines decades of Treasury yield data
- **Recession Correlation**: Quantifies the relationship between yield curve inversion and recessions
- **Lead Time Analysis**: Determines how far in advance inversions predict recessions
- **Visualization**: Creates informative charts for trend analysis
- **Statistical Metrics**: Provides precision, recall, and F1 scores for prediction accuracy

## Economic Theory

### The Yield Curve and Economic Cycles

The Treasury yield curve represents the interest rates on U.S. government bonds of different maturities. It is a critical indicator of market expectations about future economic conditions.

#### Normal vs. Inverted Yield Curve

- **Normal Curve**: Long-term rates > Short-term rates. Investors demand higher yields for longer commitments due to inflation risk.
- **Inverted Curve**: Long-term rates < Short-term rates. Indicates market expectations of declining future interest rates and economic slowdown.

#### Why Inversion Predicts Recessions

1. **Market Expectations**: An inverted curve suggests investors expect the Federal Reserve to cut rates in response to slowing economic growth
2. **Credit Constraints**: Higher short-term rates relative to long-term rates can reduce business borrowing and investment
3. **Investor Sentiment**: Inversion reflects pessimism about near-term economic prospects
4. **Historical Accuracy**: In the past 60+ years, nearly every recession was preceded by yield curve inversion

#### The Lead Time

- Yield curve inversions typically precede recessions by **6-24 months**
- This lag allows investors and policymakers time to prepare
- The exact timing varies depending on economic conditions and policy responses

### Key Economic Indicators Used

**10-Year minus 2-Year Spread**:
- Most commonly used in recession prediction models
- Captures medium to long-term economic sentiment
- Data is widely available and reliable

## Installation Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries

The script requires the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn fredapi scikit-learn
```

### Individual Package Installation

```bash
# Data retrieval and manipulation
pip install pandas>=1.3.0
pip install numpy>=1.20.0

# Data sources
pip install fredapi

# Visualization
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# Statistical analysis
pip install scikit-learn>=0.24.0
```

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/JConz801/fine-ants.git
cd fine-ants
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import pandas, matplotlib; print('All dependencies installed successfully')"
```

## Usage

### Basic Syntax

```bash
python treasury_spread_recession_prediction.py [options]
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--start-date` | Start date for analysis (YYYY-MM-DD) | 1980 |
| `--end-date` | End date for analysis (YYYY-MM-DD) | Today |
| `--long-term` | Long-term Treasury maturity in years | 10 |
| `--short-term` | Short-term Treasury maturity in years | 2 |
| `--inversion-threshold` | Spread threshold for inversion (default: 0) | 0.0 |
| `--output-dir` | Directory for output files | ./output |
| `--show-plots` | Display plots (True/False) | True |
| `--verbose` | Detailed output (True/False) | False |

### Examples

#### Basic Usage
```bash
python treasury_spread_recession_prediction.py
```

#### With Custom Date Range
```bash
python treasury_spread_recession_prediction.py --start-date 2015-01-01 --end-date 2024-12-31
```

#### Different Treasury Spreads
```bash
python treasury_spread_recession_prediction.py --long-term 10 --short-term 2
```

#### Specify Output Directory
```bash
python treasury_spread_recession_prediction.py --output-dir ./analysis_results
```

## Example Usage

### Scenario 1: Recent Recession Analysis

```bash
python treasury_spread_recession_prediction.py \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --verbose True
```

This analyzes the period around the COVID-19 pandemic and subsequent recovery.

### Scenario 2: Full Historical Analysis

```bash
python treasury_spread_recession_prediction.py \
  --start-date 1980-01-01 \
  --end-date 2024-12-31 \
  --output-dir ./historical_analysis
```

This provides a comprehensive look at the late 20th (since 1980) and 21st century, including multiple recession cycles.

### Scenario 3: Custom Spread Configuration

```bash
python treasury_spread_recession_prediction.py \
  --long-term 10 \
  --short-term 2 \
  --inversion-threshold -0.25 \
  --show-plots True
```

This uses a custom inversion threshold of -25 basis points instead of the default 0.

## Understanding the Results

### Output Metrics

#### Spread Data
- **Current Spread**: The most recent 10Y-2Y spread value
- **Historical High**: Maximum spread during the analysis period
- **Historical Low**: Minimum spread during the analysis period
- **Mean Spread**: Average spread over the analysis period

#### Recession Prediction Metrics

**Sensitivity (True Positive Rate)**
- Percentage of actual recessions preceded by inversion signals
- Higher is better; 100% means all recessions were predicted
- Formula: TP / (TP + FN)

**Specificity (True Negative Rate)**
- Percentage of non-recession periods correctly identified
- Higher is better; indicates fewer false alarms
- Formula: TN / (TN + FP)

**Precision**
- Of all inversion signals, what percentage preceded recessions?
- Important for avoiding false alarms
- Formula: TP / (TP + FP)

**F1 Score**
- Harmonic mean of precision and recall
- Balanced measure of prediction accuracy
- Range: 0 to 1 (higher is better)

#### Lead Time Analysis
- **Average Lead Time**: Typical months between inversion and recession start
- **Min/Max Lead Time**: Range of prediction windows historically observed

### Interpreting Charts

#### Spread vs. Recession Chart
- **Blue Line**: Treasury yield spread over time
- **Red Shaded Areas**: NBER-defined recession periods
- **Green Dashed Line**: Inversion threshold (typically 0)

**Interpretation**:
- When spread crosses below the green line during normal times, recession risk increases
- Most recessions are preceded by the spread becoming negative
- Current position relative to the line indicates near-term recession risk

#### Recession Probability Distribution
- **X-axis**: Months since inversion
- **Y-axis**: Probability of recession occurring
- Shows how likelihood changes as time passes after inversion

**Interpretation**:
- Peak probability typically occurs 6-18 months after inversion
- Useful for timing economic policy and investment decisions

### Example Results Interpretation

```
===== TREASURY SPREAD RECESSION PREDICTION RESULTS =====

Current Metrics:
  Current Spread: -0.35%
  Status: INVERTED (Recession Risk: HIGH)
  
Historical Analysis (2010-2024):
  Sensitivity: 0.95
  Specificity: 0.87
  Precision: 0.92
  F1 Score: 0.93
  
Lead Time Analysis:
  Average Lead Time: 14 months
  Min Lead Time: 6 months
  Max Lead Time: 24 months
  
Interpretation:
  - The yield curve is currently inverted
  - Historical model predicts recession with 95% accuracy
  - Based on historical patterns, expect recession within 6-24 months
  - Current conditions suggest ELEVATED RECESSION RISK
```

**What This Means**:
1. The negative spread indicates market pessimism
2. The high sensitivity (95%) shows the model caught almost all past recessions
3. The precision (92%) means inversion signals are reliable
4. The 14-month average lead time suggests monitoring for 1-2 years

## Output Files

The script generates the following output files:

### CSV Files

**treasury_spread_data.csv**
- Contains all calculated spreads and analysis data
- Columns: Date, 10Y_Rate, 2Y_Rate, Spread, Is_Recession, Signal

**recession_predictions.csv**
- Detailed predictions and metrics
- Columns: Date, Spread, Predicted_Recession_Prob, Signal_Given

### Visualization Files

**spread_vs_recession.png**
- Main chart showing spread trends and recession periods
- Useful for presentations and reports

**recession_probability_distribution.png**
- Distribution of recession probability after inversions
- Shows typical lead time to recession

**spread_statistics.png**
- Statistical summary of spreads
- Includes histograms and distributions

## Technical Details

### Data Sources

1. **Treasury Yields**
   - Source: Federal Reserve Economic Data (FRED) API
   - Frequency: Daily observations
   - Reliability: Highly accurate

2. **Recession Dates**
   - Source: NBER Business Cycle Dating Committee
   - Frequency: As determined by NBER
   - Authority: Official U.S. recession reference

### Calculations

**Spread Calculation**:
```
Spread = Rate_Long - Rate_Short
```

**Signal Detection**:
```
Inversion Signal = 1 if Spread < Threshold else 0
```

**Lead Time**:
```
Lead Time = Days_from_Inversion_to_Recession_Start / 30
```

### Algorithm Components

1. **Data Retrieval**: Uses 'Fred'
2. **Data Cleaning**: Handles missing values and outliers
3. **Statistical Analysis**: Uses scikit-learn for metrics
4. **Visualization**: Uses matplotlib and seaborn

## Limitations

### Known Limitations

1. **Historical Bias**: Past performance doesn't guarantee future results
   - Economic structures change over time
   - Policy responses evolve
   - Market behavior can differ

2. **Lead Time Variability**: Prediction window can range from 6-24 months
   - Timing remains unpredictable
   - Useful for risk assessment but not precise scheduling

3. **False Positives**: Inversions don't always precede recessions
   - ~5-10% of inversions don't lead to recessions
   - Especially true for brief, shallow inversions

4. **Data Limitations**:
   - Historical data only goes back to 1950s-1960s for most series
   - Recent institutional changes may affect patterns
   - International effects not captured

5. **Single Indicator**: This should not be used alone
   - Combine with other recession indicators
   - Consider economic fundamentals
   - Monitor multiple signals

### Best Practices

1. **Combine Indicators**: Use alongside:
   - Credit spreads
   - Leading Economic Index
   - Unemployment trends
   - Consumer confidence

2. **Monitor Continuously**: Check quarterly or monthly, not just once

3. **Consider Context**: Evaluate:
   - Federal Reserve policy stance
   - Inflation trends
   - Corporate earnings
   - Labor market strength

4. **Risk Management**: Use results to inform:
   - Portfolio allocation
   - Hedging strategies
   - Cash reserves
   - Business planning

## References

### Academic Literature

1. **Estrella, A., & Mishkin, F. S. (1998)**
   - "Predicting U.S. Recessions: Financial Variables as Leading Indicators"
   - Review of Economics and Statistics, 80(1), 45-61

2. **Ang, A., Bekaert, G., & Wei, M. (2007)**
   - "The term structure of real rates and expected inflation"
   - Journal of Finance, 62(2), 797-830

3. **Wheelock, D. C., & Wohar, M. E. (2009)**
   - "Can the term spread predict GDP growth and recessions? A survey of the literature"
   - Federal Reserve Bank of St. Louis Review, 91(5 Part 2), 419-40

### Official Sources

- **NBER Business Cycle Dating**: https://www.nber.org/cycles.html
- **Federal Reserve Economic Data (FRED)**: https://fred.stlouisfed.org/
- **U.S. Treasury**: https://www.treasury.gov/resource-center/data-chart-center/interest-rates

### Data Sources

- **yfinance**: Yahoo Finance data retrieval
- **pandas_datareader**: FRED and other sources
- **NBER**: Official recession dates

## Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Improve documentation

## License

This project is part of the fine-ants repository. Please refer to the main repository LICENSE file for licensing details.

## Disclaimer

This tool is provided for educational and informational purposes only. It should not be considered investment advice. Past performance is not indicative of future results. Economic predictions are inherently uncertain and subject to various risks and contingencies. Users should conduct their own research and consult with financial advisors before making investment decisions based on this analysis.

---

**Last Updated**: February 15, 2026
**Maintained By**: JConz801
**Repository**: https://github.com/JConz801/fine-ants
