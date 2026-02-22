"""
ClimateGuard API Server
Flask backend for the ClimateGuard web application
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading, webbrowser, time, os
from data_loader import load_climate_data, get_sample_regions
from data_preprocessor import DataPreprocessor
from predictor import ClimateRiskPredictor, generate_forecast
import numpy as np

CWD = os.path.abspath(os.path.dirname(__file__))
# Serve static frontend files from project root so backend and frontend share the same origin
app = Flask(__name__, static_folder=CWD, static_url_path='')
CORS(app)  # Enable CORS for frontend

# Initialize components
preprocessor = DataPreprocessor()
predictor = ClimateRiskPredictor()

# Load and train model on startup
print("Loading training data...")
training_data = load_climate_data(days=30)
training_data = preprocessor.process_pipeline(training_data)
print("Training prediction models...")
predictor.train(training_data)
print("Models ready!")

# Store regions
regions = get_sample_regions()


@app.route('/')
def serve_frontend():
    """Serve the frontend index file from project root."""
    return send_from_directory(CWD, 'index.html')

@app.route('/api', methods=['GET'])
def api_home():
    """API home info endpoint"""
    return jsonify({
        'service': 'ClimateGuard API',
        'version': '1.0.0',
        'status': 'operational'
    })


@app.route('/regions', methods=['GET'])
def get_regions():
    """Get all monitored regions"""
    return jsonify({
        'regions': regions,
        'count': len(regions)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Generate climate risk predictions"""
    data = request.json
    
    region_ids = data.get('region_ids', [1])
    disaster_type = data.get('disaster_type', 'flood')
    forecast_days = data.get('forecast_days', 7)
    
    # Generate predictions for each region
    predictions = []
    
    for region_id in region_ids:
        # Find region
        region = next((r for r in regions if r['id'] == region_id), None)
        if not region:
            continue
        
        # Load recent data for this region
        recent_data = load_climate_data(region_id=region_id, days=7)
        recent_data = preprocessor.process_pipeline(recent_data)
        
        # Generate forecast
        forecasts = generate_forecast(predictor, recent_data, days=forecast_days)
        
        # Determine overall risk
        if disaster_type == 'flood':
            max_prob = max(f['flood_probability'] for f in forecasts)
            overall_risk = predictor.classify_risk(max_prob)
        else:
            max_prob = max(f['heatwave_probability'] for f in forecasts)
            overall_risk = predictor.classify_risk(max_prob)
        
        predictions.append({
            'region_id': region_id,
            'region_name': region['name'],
            'forecasts': forecasts,
            'overall_risk': overall_risk
        })
    
    return jsonify({
        'disaster_type': disaster_type,
        'forecast_days': forecast_days,
        'predictions': predictions,
        'timestamp': training_data['timestamp'].max().isoformat()
    })


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get active climate alerts"""
    alerts = []
    
    # Generate alerts for regions with high risk
    for region in regions:
        # Load recent data
        recent_data = load_climate_data(region_id=region['id'], days=1)
        recent_data = preprocessor.process_pipeline(recent_data)
        
        # Check flood risk
        flood_prob = predictor.predict_flood(recent_data).mean()
        if flood_prob > 0.5:
            alerts.append({
                'region_id': region['id'],
                'region_name': region['name'],
                'type': 'flood',
                'severity': predictor.classify_risk(flood_prob),
                'probability': float(flood_prob),
                'message': f'Elevated flood risk detected in {region["name"]}'
            })
        
        # Check heatwave risk
        heatwave_prob = predictor.predict_heatwave(recent_data).mean()
        if heatwave_prob > 0.5:
            alerts.append({
                'region_id': region['id'],
                'region_name': region['name'],
                'type': 'heatwave',
                'severity': predictor.classify_risk(heatwave_prob),
                'probability': float(heatwave_prob),
                'message': f'Elevated heatwave risk detected in {region["name"]}'
            })
    
    return jsonify({
        'alerts': alerts,
        'count': len(alerts)
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.is_trained,
        'regions_monitored': len(regions)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌍 ClimateGuard API Server Starting...")
    print("="*60)
    print(f"Monitoring {len(regions)} regions")
    print("Server will run on: http://localhost:5000")
    print("="*60 + "\n")
    
    # Open the default browser shortly after the server starts
    def _open_browser():
        time.sleep(1.2)
        webbrowser.open(f'http://localhost:5000')

    threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=True)
