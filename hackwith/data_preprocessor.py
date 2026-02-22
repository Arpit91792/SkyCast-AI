"""
Data Preprocessing Module
Cleans, transforms and prepares data for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Preprocess climate data for machine learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def clean_data(self, df):
        """
        Clean the data by handling missing values and outliers
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Remove outliers (values beyond 3 std deviations)
        for col in ['temperature', 'rainfall', 'humidity']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
        
        return df
    
    def create_features(self, df):
        """
        Create additional features from raw data
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Rolling averages (24-hour window)
        df['temp_rolling_24h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
        df['rain_rolling_24h'] = df['rainfall'].rolling(window=24, min_periods=1).sum()
        df['humidity_rolling_24h'] = df['humidity'].rolling(window=24, min_periods=1).mean()
        
        # Temperature change rate
        df['temp_change'] = df['temperature'].diff()
        
        # Rain intensity indicator
        df['rain_intensity'] = pd.cut(df['rainfall'], 
                                      bins=[0, 2.5, 10, 50, 1000],
                                      labels=['light', 'moderate', 'heavy', 'extreme'])
        df['rain_intensity'] = df['rain_intensity'].cat.codes
        
        return df
    
    def normalize_features(self, df, columns=None):
        """
        Normalize features to 0-1 range
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize
        
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = ['temperature', 'rainfall', 'humidity', 'aqi']
        
        available_cols = [col for col in columns if col in df.columns]
        
        if not self.fitted:
            df[available_cols] = self.scaler.fit_transform(df[available_cols])
            self.fitted = True
        else:
            df[available_cols] = self.scaler.transform(df[available_cols])
        
        return df
    
    def prepare_sequences(self, df, seq_length=24, target_col='temperature'):
        """
        Create sequences for time series prediction
        
        Args:
            df: Input DataFrame
            seq_length: Length of input sequence
            target_col: Column to predict
        
        Returns:
            X (sequences), y (targets)
        """
        feature_cols = ['temperature', 'rainfall', 'humidity', 'aqi']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X, y = [], []
        
        for i in range(len(df) - seq_length):
            X.append(df[available_cols].iloc[i:i+seq_length].values)
            y.append(df[target_col].iloc[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def process_pipeline(self, df):
        """
        Run complete preprocessing pipeline
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Processed DataFrame
        """
        print("Cleaning data...")
        df = self.clean_data(df)
        
        print("Creating features...")
        df = self.create_features(df)
        
        print("Normalizing features...")
        df = self.normalize_features(df)
        
        return df


if __name__ == "__main__":
    from data_loader import load_climate_data
    
    # Load sample data
    df = load_climate_data(days=7)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.process_pipeline(df)
    
    print(f"\nOriginal columns: {df.shape[1]}")
    print(f"Processed columns: {processed_df.shape[1]}")
    print(f"\nNew features: {set(processed_df.columns) - set(df.columns)}")
    print(f"\nFirst few rows:\n{processed_df.head()}")