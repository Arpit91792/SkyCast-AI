"""
Climate Risk Predictor
Simple prediction model for climate risks
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ClimateRiskPredictor:
    """Predict climate risks (flood, heatwave) using Random Forest"""
    
    def __init__(self):
        self.flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.heatwave_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def create_risk_labels(self, df):
        """
        Create risk labels from climate data
        
        Args:
            df: DataFrame with climate data
        
        Returns:
            DataFrame with risk labels added
        """
        df = df.copy()
        
        # Flood risk: high rainfall + high rolling rainfall
        df['flood_risk'] = 0
        df.loc[(df['rainfall'] > df['rainfall'].quantile(0.8)) | 
               (df.get('rain_rolling_24h', 0) > df.get('rain_rolling_24h', pd.Series([0])).quantile(0.8)), 
               'flood_risk'] = 1
        
        # Heatwave risk: high temperature + sustained heat
        df['heatwave_risk'] = 0
        df.loc[(df['temperature'] > df['temperature'].quantile(0.85)) & 
               (df.get('temp_rolling_24h', 0) > df.get('temp_rolling_24h', pd.Series([0])).quantile(0.8)), 
               'heatwave_risk'] = 1
        
        return df
    
    def train(self, df):
        """
        Train the prediction models
        
        Args:
            df: Training DataFrame with features and risk labels
        """
        # Create risk labels if not present
        if 'flood_risk' not in df.columns:
            df = self.create_risk_labels(df)
        
        # Select features
        feature_cols = ['temperature', 'rainfall', 'humidity', 'aqi',
                       'temp_rolling_24h', 'rain_rolling_24h', 
                       'humidity_rolling_24h', 'hour', 'day_of_week', 'month']
        
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Train flood model
        y_flood = df['flood_risk']
        if len(y_flood.unique()) > 1:
            self.flood_model.fit(X, y_flood)
            print(f"Flood model trained on {len(X)} samples")
        
        # Train heatwave model
        y_heatwave = df['heatwave_risk']
        if len(y_heatwave.unique()) > 1:
            self.heatwave_model.fit(X, y_heatwave)
            print(f"Heatwave model trained on {len(X)} samples")
        
        self.is_trained = True
        self.feature_cols = available_features
    
    def predict_flood(self, df):
        """
        Predict flood risk
        
        Args:
            df: DataFrame with features
        
        Returns:
            Array of probabilities [0-1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[self.feature_cols].fillna(0)
        probabilities = self.flood_model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def predict_heatwave(self, df):
        """
        Predict heatwave risk
        
        Args:
            df: DataFrame with features
        
        Returns:
            Array of probabilities [0-1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[self.feature_cols].fillna(0)
        probabilities = self.heatwave_model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def classify_risk(self, probability):
        """
        Classify risk level based on probability
        
        Args:
            probability: Risk probability [0-1]
        
        Returns:
            Risk level: 'safe', 'warning', 'danger', 'extreme'
        """
        if probability < 0.3:
            return 'safe'
        elif probability < 0.6:
            return 'warning'
        elif probability < 0.85:
            return 'danger'
        else:
            return 'extreme'
    
    def predict_with_classification(self, df, disaster_type='flood'):
        """
        Predict and classify risk
        
        Args:
            df: DataFrame with features
            disaster_type: 'flood' or 'heatwave'
        
        Returns:
            DataFrame with predictions and classifications
        """
        df = df.copy()
        
        if disaster_type == 'flood':
            df['probability'] = self.predict_flood(df)
        else:
            df['probability'] = self.predict_heatwave(df)
        
        df['risk_level'] = df['probability'].apply(self.classify_risk)
        
        return df


def generate_forecast(predictor, current_data, days=7):
    """
    Generate multi-day forecast
    
    Args:
        predictor: Trained ClimateRiskPredictor
        current_data: Current climate data
        days: Number of days to forecast
    
    Returns:
        List of predictions for each day
    """
    from datetime import datetime, timedelta
    
    forecasts = []
    
    for day in range(days):
        forecast_date = datetime.now() + timedelta(days=day+1)
        
        # Simple forecast: use recent averages with some variation
        temp = current_data['temperature'].mean() + np.random.randn() * 2
        rain = max(0, current_data['rainfall'].mean() + np.random.randn() * 5)
        humidity = current_data['humidity'].mean() + np.random.randn() * 5
        aqi = int(current_data['aqi'].mean() + np.random.randn() * 10)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'temperature': [temp],
            'rainfall': [rain],
            'humidity': [humidity],
            'aqi': [aqi],
            'temp_rolling_24h': [temp],
            'rain_rolling_24h': [rain * 24],
            'humidity_rolling_24h': [humidity],
            'hour': [12],
            'day_of_week': [forecast_date.weekday()],
            'month': [forecast_date.month]
        })
        
        # Predict
        flood_prob = predictor.predict_flood(forecast_df)[0]
        heatwave_prob = predictor.predict_heatwave(forecast_df)[0]
        
        forecasts.append({
            'date': forecast_date.strftime('%Y-%m-%d'),
            'flood_probability': round(flood_prob, 3),
            'flood_risk': predictor.classify_risk(flood_prob),
            'heatwave_probability': round(heatwave_prob, 3),
            'heatwave_risk': predictor.classify_risk(heatwave_prob),
            'temperature': round(temp, 1),
            'rainfall': round(rain, 1)
        })
    
    return forecasts


if __name__ == "__main__":
    from data_loader import load_climate_data
    from data_preprocessor import DataPreprocessor
    
    # Load and preprocess data
    df = load_climate_data(days=30)
    preprocessor = DataPreprocessor()
    df = preprocessor.process_pipeline(df)
    
    # Train predictor
    predictor = ClimateRiskPredictor()
    predictor.train(df)
    
    # Generate 7-day forecast
    forecasts = generate_forecast(predictor, df, days=7)
    
    print("\n7-Day Climate Risk Forecast:")
    print("=" * 80)
    for forecast in forecasts:
        print(f"\nDate: {forecast['date']}")
        print(f"  Flood: {forecast['flood_probability']:.1%} ({forecast['flood_risk']})")
        print(f"  Heatwave: {forecast['heatwave_probability']:.1%} ({forecast['heatwave_risk']})")
        print(f"  Expected Temp: {forecast['temperature']}°C, Rain: {forecast['rainfall']}mm")