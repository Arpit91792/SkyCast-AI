# 🌍 ClimateGuard - AI-Powered Climate Risk Prediction Platform

ClimateGuard is an intelligent climate risk prediction system that uses machine learning to forecast floods and heatwaves for monitored regions.

## 🚀 Features

- **Real-time Climate Predictions**: Forecast flood and heatwave risks up to 14 days ahead
- **Multi-Region Monitoring**: Track climate risks across multiple geographic regions
- **Interactive Dashboard**: Visualize predictions, alerts, and statistics
- **Risk Classification**: Automatic classification into Safe, Warning, Danger, and Extreme levels
- **Alert System**: Get notified about high-risk climate events

## 📋 Requirements

- Python 3.13+
- Modern web browser (Chrome, Firefox, Edge)

## 🛠️ Installation

All dependencies are already installed. The project uses:
- Flask & Flask-CORS (Backend API)
- scikit-learn (Machine Learning)
- pandas & numpy (Data Processing)
- PyTorch & torch-geometric (Deep Learning - STGCN model)

## 🎯 Running the Application

### 1. Start the Backend API Server
```bash
python api_server.py
```
The API will run on http://localhost:5000

### 2. Start the Frontend Server
```bash
python -m http.server 8080
```
The web interface will be available at http://localhost:8080

### 3. Open in Browser
Navigate to http://localhost:8080 in your web browser

## 📁 Project Structure

```
climateguard/
├── api_server.py           # Flask API backend
├── data_loader.py          # Data loading utilities
├── data_preprocessor.py    # Data preprocessing pipeline
├── predictor.py            # ML prediction models
├── climateguard_stcgn.py   # STGCN deep learning model
├── climateguard_api.py     # FastAPI alternative (advanced)
├── index.html              # Frontend interface
├── style.css               # Styling
├── app.js                  # Frontend logic
└── README.md               # This file
```

## 🔧 API Endpoints

- `GET /` - API status
- `GET /regions` - List all monitored regions
- `POST /predict` - Generate climate predictions
- `GET /alerts` - Get active climate alerts
- `GET /health` - Health check

## 💡 Usage

1. **View Dashboard**: See overall statistics and recent predictions
2. **Generate Predictions**: 
   - Select disaster type (Flood/Heatwave)
   - Choose forecast period (1-14 days)
   - Select regions to monitor
   - Click "Get Predictions"
3. **Check Alerts**: View active climate warnings
4. **Browse Regions**: See all monitored geographic areas

## 🧪 Testing the Models

Run individual components:

```bash
# Test data loader
python data_loader.py

# Test preprocessor
python data_preprocessor.py

# Test predictor
python predictor.py

# Test STGCN model
python climateguard_stcgn.py
```

## 🌐 Current Status

✅ Backend API Server: Running on http://localhost:5000
✅ Frontend Server: Running on http://localhost:8080
✅ ML Models: Trained and ready
✅ Monitoring: 5 regions

## 📊 Models

- **Random Forest Classifier**: Fast predictions for flood and heatwave risks
- **STGCN (Spatio-Temporal Graph Convolutional Network)**: Advanced deep learning model for spatial-temporal climate patterns

## 🎨 Features in the Web Interface

- Real-time statistics dashboard
- Interactive prediction form
- Color-coded risk levels
- Detailed forecast tables
- Active alerts monitoring
- Region information cards

## 🔮 Future Enhancements

- Historical data visualization
- Email/SMS alert notifications
- Mobile app integration
- More disaster types (drought, storms)
- Integration with real weather APIs
- User authentication system

## 📝 Notes

- The current implementation uses simulated climate data for demonstration
- In production, integrate with real weather APIs (OpenWeatherMap, NOAA, etc.)
- Models are trained on synthetic data - retrain with real historical data for accuracy

## 🤝 Contributing

This is a demonstration project. Feel free to extend and improve it!

---

**ClimateGuard** - Predicting Climate Risks Before They Strike 🌍⚡
