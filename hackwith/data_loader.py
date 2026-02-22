"""
Data Loader Module
Loads climate data from CSV files or generates sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_climate_data(filepath=None, region_id=1, days=30):
    """
    Load climate data from file or generate sample data
    
    Args:
        filepath: Path to CSV file (optional)
        region_id: ID of the region
        days: Number of days of data to generate
    
    Returns:
        DataFrame with climate data
    """
    if filepath:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        return df
    
    # Generate sample data
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days*24, freq='h')
    
    data = {
        'timestamp': dates,
        'region_id': region_id,
        'temperature': np.random.randn(len(dates)) * 5 + 25,  # Mean 25°C
        'rainfall': np.abs(np.random.randn(len(dates)) * 10),  # mm
        'humidity': np.clip(np.random.randn(len(dates)) * 10 + 60, 0, 100),  # %
        'aqi': np.random.randint(20, 150, len(dates)),  # Air Quality Index
    }
    
    df = pd.DataFrame(data)
    return df


def get_sample_regions():
    """Get list of sample regions"""
    return [
        {'id': 1, 'name': 'Central District', 'lat': 21.25, 'lon': 81.63},
        {'id': 2, 'name': 'North Zone', 'lat': 21.30, 'lon': 81.60},
        {'id': 3, 'name': 'South Zone', 'lat': 21.20, 'lon': 81.65},
        {'id': 4, 'name': 'East Zone', 'lat': 21.25, 'lon': 81.70},
        {'id': 5, 'name': 'West Zone', 'lat': 21.25, 'lon': 81.56},
    ]


def save_climate_data(df, filepath):
    """Save climate data to CSV"""
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


if __name__ == "__main__":
    # Test the data loader
    df = load_climate_data(days=7)
    print(f"Loaded {len(df)} records")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Save sample data
    save_climate_data(df, "sample_climate_data.csv")