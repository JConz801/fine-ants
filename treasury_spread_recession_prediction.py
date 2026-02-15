"""
Treasury Spread Recession Prediction Analysis

This module analyzes the predictive power of the ten year - two year U.S. Treasury
interest rate spread for identifying recessions. The inverted yield curve (negative
spread) has historically been a reliable indicator of economic downturns.

Author: JConz801
Date: 2026-01-05
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
import json
import os
from fredapi import Fred

warnings.filterwarnings('ignore')


def load_api_key_from_notebook(notebook_path: str = 'key_f.ipynb') -> str:
    """
    Load the FRED API key from key_f.ipynb notebook.
    
    Args:
        notebook_path: Path to the notebook file containing the api_key
        
    Returns:
        The API key string
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, notebook_path)
    
    with open(full_path, 'r') as f:
        notebook = json.load(f)
    
    # Extract api_key from the notebook cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'api_key' in source and '=' in source:
                # Extract the string value between quotes
                import re
                match = re.search(r"api_key\s*=\s*['\"]([^'\"]+)['\"]", source)
                if match:
                    return match.group(1)
    
    raise ValueError(f"Could not find api_key in {notebook_path}")


# Load API key from notebook
api_key = load_api_key_from_notebook()

class TreasurySpreadRecessionPredictor:
    """
    Analyzes the relationship between the 10Y-2Y Treasury spread and recessions.
    
    The yield curve spread is calculated as: 10-Year Treasury Rate - 2-Year Treasury Rate
    Negative spreads (inverted yield curve) typically precede recessions.
    """
    
    def __init__(self):
        """Initialize the predictor with historical data."""
        self.spread_data = None
        self.recession_data = None
        self.analysis_results = {}
    
    def load_sample_data(self, api_key: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load historical data from FRED API.
        
        Args:
            api_key: FRED API key. If None, uses module-level api_key variable.
            
        Returns:
            Tuple of (spread_df, recession_df)
        """
        # Get API key from parameter or use module-level default
        if api_key is None:
            api_key = globals()['api_key']
        
        # Initialize FRED API
        fred = Fred(api_key=api_key)
        
        # Fetch 10Y-2Y spread data (T10Y2Y series)
        print("Fetching Treasury spread data from FRED...")
        spread_series = fred.get_series('T10Y2Y', observation_start='1980-01-01')
        
        spread_df = pd.DataFrame({
            'date': spread_series.index,
            'spread_bps': spread_series.values * 100  # Convert from percentage to basis points
        })
        spread_df = spread_df.reset_index(drop=True)
        
        # Define historical recession periods (NBER official dates)
        recession_periods = [
            ('1980-01-01', '1980-07-01'),   # 1980 Recession
            ('1981-07-01', '1982-11-01'),   # Early 1980s Recession
            ('1990-07-01', '1991-03-01'),   # 1990-1991 Recession
            ('2001-03-01', '2001-11-01'),   # 2001 Recession
            ('2007-12-01', '2009-06-01'),   # 2008-2009 Financial Crisis
            ('2020-02-01', '2020-04-01'),   # COVID-19 Recession
        ]
        
        recession_list = []
        for start, end in recession_periods:
            recession_list.append({
                'start_date': pd.to_datetime(start),
                'end_date': pd.to_datetime(end),
                'recession_name': f"Recession {start[:4]}"
            })
        
        recession_df = pd.DataFrame(recession_list)
        
        print(f"Loaded {len(spread_df)} months of Treasury spread data")
        print(f"Data range: {spread_df['date'].min().date()} to {spread_df['date'].max().date()}")
        
        return spread_df, recession_df
    
    def preprocess_data(self, spread_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess spread data and calculate technical indicators.
        
        Args:
            spread_df: DataFrame with spread data
            
        Returns:
            Preprocessed DataFrame with additional features
        """
        df = spread_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate rolling statistics
        df['spread_ma6'] = df['spread_bps'].rolling(window=6).mean()
        df['spread_ma12'] = df['spread_bps'].rolling(window=12).mean()
        df['spread_volatility'] = df['spread_bps'].rolling(window=12).std()
        
        # Calculate changes
        df['spread_change'] = df['spread_bps'].diff()
        df['spread_momentum'] = df['spread_bps'].rolling(window=3).mean().diff()
        
        # Inversion indicator (1 if inverted, 0 otherwise)
        df['inverted'] = (df['spread_bps'] < 0).astype(int)
        
        # Months in inversion
        df['inversion_streak'] = df['inverted'].groupby(
            (df['inverted'] != df['inverted'].shift()).cumsum()
        ).cumcount() + 1
        df.loc[df['inverted'] == 0, 'inversion_streak'] = 0
        
        return df
    
    def analyze_inversion_predictive_power(
        self, 
        spread_df: pd.DataFrame, 
        recession_df: pd.DataFrame,
        lead_months: int = 12
    ) -> Dict:
        """
        Analyze how well yield curve inversions predict recessions.
        
        Args:
            spread_df: DataFrame with spread data
            recession_df: DataFrame with recession periods
            lead_months: Months in advance to check for inversion signal
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'total_recession_periods': len(recession_df),
            'inversions_before_recession': 0,
            'false_positives': 0,
            'recession_prediction_accuracy': 0.0,
            'average_inversion_lead_time': [],
            'inversion_detection_rate': 0.0
        }
        
        spread_df = spread_df.copy()
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        
        for _, recession in recession_df.iterrows():
            recession_start = recession['start_date']
            lookback_period = recession_start - timedelta(days=lead_months * 30)
            
            # Check for inversions before recession
            prior_inversions = spread_df[
                (spread_df['date'] >= lookback_period) & 
                (spread_df['date'] < recession_start) &
                (spread_df['spread_bps'] < 0)
            ]
            
            if len(prior_inversions) > 0:
                results['inversions_before_recession'] += 1
                lead_time = (recession_start - prior_inversions['date'].min()).days / 30
                results['average_inversion_lead_time'].append(lead_time)
        
        # Calculate metrics
        if len(results['average_inversion_lead_time']) > 0:
            results['average_inversion_lead_time'] = np.mean(
                results['average_inversion_lead_time']
            )
        
        if results['total_recession_periods'] > 0:
            results['inversion_detection_rate'] = (
                results['inversions_before_recession'] / 
                results['total_recession_periods'] * 100
            )
        
        return results
    
    def calculate_confusion_matrix(
        self, 
        spread_df: pd.DataFrame,
        recession_df: pd.DataFrame,
        inversion_threshold: float = 0.0,
        lead_months: int = 12
    ) -> Dict:
        """
        Calculate confusion matrix metrics for the inversion signal.
        
        Args:
            spread_df: DataFrame with spread data
            recession_df: DataFrame with recession periods
            inversion_threshold: Threshold for inversion (default 0)
            lead_months: Months to look ahead
            
        Returns:
            Dictionary with confusion matrix metrics
        """
        # Create binary recession indicator
        spread_df = spread_df.copy()
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df['in_recession'] = 0
        
        for _, recession in recession_df.iterrows():
            mask = (
                (spread_df['date'] >= recession['start_date']) &
                (spread_df['date'] <= recession['end_date'])
            )
            spread_df.loc[mask, 'in_recession'] = 1
        
        # Create lagged inversion signal
        spread_df['inverted'] = (spread_df['spread_bps'] < inversion_threshold).astype(int)
        spread_df['inversion_signal'] = spread_df['inverted'].shift(lead_months)
        
        # Remove NaN values
        analysis_df = spread_df.dropna()
        
        # Calculate metrics
        tp = ((analysis_df['inversion_signal'] == 1) & (analysis_df['in_recession'] == 1)).sum()
        tn = ((analysis_df['inversion_signal'] == 0) & (analysis_df['in_recession'] == 0)).sum()
        fp = ((analysis_df['inversion_signal'] == 1) & (analysis_df['in_recession'] == 0)).sum()
        fn = ((analysis_df['inversion_signal'] == 0) & (analysis_df['in_recession'] == 1)).sum()
        
        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'f1_score': 0.0
        }
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (
                (metrics['precision'] * metrics['recall']) / 
                (metrics['precision'] + metrics['recall'])
            )
        
        return metrics
    
    def run_analysis(self, api_key: str = None) -> Dict:
        """
        Execute the full recession prediction analysis.
        
        Args:
            api_key: FRED API key. If None, uses FRED_API_KEY environment variable.
        
        Returns:
            Dictionary with complete analysis results
        """
        # Load data
        spread_df, recession_df = self.load_sample_data(api_key=api_key)
        self.spread_data = spread_df
        self.recession_data = recession_df
        
        # Preprocess
        processed_data = self.preprocess_data(spread_df)
        
        # Run analyses
        inversion_analysis = self.analyze_inversion_predictive_power(
            spread_df, recession_df, lead_months=12
        )
        
        confusion_metrics = self.calculate_confusion_matrix(
            spread_df, recession_df, inversion_threshold=0.0, lead_months=12
        )
        
        self.analysis_results = {
            'inversion_analysis': inversion_analysis,
            'confusion_matrix': confusion_metrics,
            'processed_data': processed_data,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        return self.analysis_results
    
    def print_summary(self):
        """Print a formatted summary of analysis results."""
        if not self.analysis_results:
            print("No analysis results. Run run_analysis() first.")
            return
        
        results = self.analysis_results
        inversion = results['inversion_analysis']
        confusion = results['confusion_matrix']
        
        print("\n" + "="*70)
        print("TREASURY SPREAD RECESSION PREDICTION ANALYSIS")
        print("="*70)
        
        print("\n--- INVERSION ANALYSIS ---")
        print(f"Total Recession Periods: {inversion['total_recession_periods']}")
        print(f"Inversions Before Recession: {inversion['inversions_before_recession']}")
        print(f"Detection Rate: {inversion['inversion_detection_rate']:.1f}%")
        if isinstance(inversion['average_inversion_lead_time'], (int, float)):
            print(f"Average Lead Time: {inversion['average_inversion_lead_time']:.1f} months")
        
        print("\n--- CONFUSION MATRIX METRICS ---")
        print(f"True Positives: {confusion['true_positives']}")
        print(f"True Negatives: {confusion['true_negatives']}")
        print(f"False Positives: {confusion['false_positives']}")
        print(f"False Negatives: {confusion['false_negatives']}")
        
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Accuracy:  {confusion['accuracy']:.2%}")
        print(f"Precision: {confusion['precision']:.2%}")
        print(f"Recall:    {confusion['recall']:.2%}")
        print(f"F1-Score:  {confusion['f1_score']:.4f}")
        
        print("\n" + "="*70)
        print(f"Analysis completed at: {results['analysis_timestamp']}")
        print("="*70 + "\n")


def main():
    """Main execution function."""
    print("Initializing Treasury Spread Recession Prediction Analysis...\n")
    
    predictor = TreasurySpreadRecessionPredictor()
    results = predictor.run_analysis()
    predictor.print_summary()
    
    return results


if __name__ == "__main__":
    analysis_results = main()
