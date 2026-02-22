"""
ClimateGuard FastAPI Backend
RESTful API for climate risk predictions and data access
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import torch

# Initialize FastAPI app
app = FastAPI(
    title="ClimateGuard API",
    description="AI-powered climate risk prediction platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Enums
class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EXTREME = "extreme"

class DisasterType(str, Enum):
    FLOOD = "flood"
    HEATWAVE = "heatwave"
    BOTH = "both"


# Request Models
class PredictionRequest(BaseModel):
    region_ids: List[int] = Field(..., description="List of region IDs to predict")
    disaster_type: DisasterType = Field(..., description="Type of disaster to predict")
    forecast_days: int = Field(7, ge=1, le=30, description="Number of days to forecast")
    
    class Config:
        schema_extra = {
            "example": {
                "region_ids": [1, 2, 3],
                "disaster_type": "flood",
                "forecast_days": 7
            }
        }


class HistoricalDataRequest(BaseModel):
    region_id: int = Field(..., description="Region ID")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    variables: Optional[List[str]] = Field(None, description="Climate variables to fetch")
    
    class Config:
        schema_extra = {
            "example": {
                "region_id": 1,
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T23:59:59",
                "variables": ["temperature", "rainfall", "humidity"]
            }
        }


class AlertSubscription(BaseModel):
    user_id: str = Field(..., description="User ID")
    region_ids: List[int] = Field(..., description="Regions to monitor")
    disaster_types: List[DisasterType] = Field(..., description="Disaster types to monitor")
    risk_threshold: RiskLevel = Field(..., description="Minimum risk level for alerts")
    notification_channels: List[str] = Field(..., description="email, sms, push")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "region_ids": [1, 2, 3],
                "disaster_types": ["flood", "heatwave"],
                "risk_threshold": "warning",
                "notification_channels": ["email", "push"]
            }
        }


# Response Models
class RegionPrediction(BaseModel):
    region_id: int
    region_name: str
    predictions: List[Dict] = Field(..., description="Daily predictions")
    overall_risk: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    timestamp: datetime
    disaster_type: DisasterType
    forecast_days: int
    regions: List[RegionPrediction]
    metadata: Dict = Field(default_factory=dict)


class HistoricalDataResponse(BaseModel):
    region_id: int
    region_name: str
    start_date: datetime
    end_date: datetime
    data: List[Dict]
    statistics: Dict


class AlertResponse(BaseModel):
    alert_id: str
    timestamp: datetime
    region_id: int
    region_name: str
    disaster_type: DisasterType
    risk_level: RiskLevel
    probability: float
    message: str
    recommendations: List[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    model_version: str


# Database models (simplified - would use SQLAlchemy in production)
class RegionDB:
    """Mock database for regions"""
    regions = {
        1: {"id": 1, "name": "Central District", "lat": 21.25, "lon": 81.63},
        2: {"id": 2, "name": "North Zone", "lat": 21.30, "lon": 81.60},
        3: {"id": 3, "name": "South Zone", "lat": 21.20, "lon": 81.65},
    }
    
    @classmethod
    def get_region(cls, region_id: int):
        return cls.regions.get(region_id)
    
    @classmethod
    def get_all_regions(cls):
        return list(cls.regions.values())


# Dependency injection
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify API key authentication"""
    # In production, check against database
    valid_keys = ["demo-key-12345", "test-key-67890"]
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# Mock ML model loader
class ModelService:
    """Service for loading and managing ML models"""
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        # In production, load actual PyTorch models
        self.models['flood'] = None  # Load flood prediction model
        self.models['heatwave'] = None  # Load heatwave prediction model
        print("Models loaded successfully")
    
    def predict_flood(self, region_ids: List[int], forecast_days: int):
        """Generate flood predictions"""
        # Mock predictions - replace with actual model inference
        predictions = []
        for region_id in region_ids:
            region = RegionDB.get_region(region_id)
            if not region:
                continue
            
            daily_predictions = []
            for day in range(forecast_days):
                # Simulate prediction
                base_prob = np.random.beta(2, 5)  # Skewed towards lower probabilities
                daily_predictions.append({
                    "date": (datetime.now() + timedelta(days=day+1)).isoformat(),
                    "probability": float(base_prob),
                    "risk_level": self._classify_risk(base_prob),
                    "temperature": round(25 + np.random.randn() * 3, 1),
                    "rainfall": round(max(0, 10 + np.random.randn() * 15), 1),
                })
            
            overall_risk = self._overall_risk(daily_predictions)
            
            predictions.append(RegionPrediction(
                region_id=region_id,
                region_name=region['name'],
                predictions=daily_predictions,
                overall_risk=overall_risk,
                confidence=0.85
            ))
        
        return predictions
    
    def predict_heatwave(self, region_ids: List[int], forecast_days: int):
        """Generate heatwave predictions"""
        predictions = []
        for region_id in region_ids:
            region = RegionDB.get_region(region_id)
            if not region:
                continue
            
            daily_predictions = []
            for day in range(forecast_days):
                base_prob = np.random.beta(2, 5)
                daily_predictions.append({
                    "date": (datetime.now() + timedelta(days=day+1)).isoformat(),
                    "probability": float(base_prob),
                    "risk_level": self._classify_risk(base_prob),
                    "max_temperature": round(35 + np.random.randn() * 5, 1),
                    "heat_index": round(40 + np.random.randn() * 6, 1),
                })
            
            overall_risk = self._overall_risk(daily_predictions)
            
            predictions.append(RegionPrediction(
                region_id=region_id,
                region_name=region['name'],
                predictions=daily_predictions,
                overall_risk=overall_risk,
                confidence=0.82
            ))
        
        return predictions
    
    @staticmethod
    def _classify_risk(probability: float) -> RiskLevel:
        """Classify risk level based on probability"""
        if probability < 0.3:
            return RiskLevel.SAFE
        elif probability < 0.6:
            return RiskLevel.WARNING
        elif probability < 0.85:
            return RiskLevel.DANGER
        else:
            return RiskLevel.EXTREME
    
    @staticmethod
    def _overall_risk(predictions: List[Dict]) -> RiskLevel:
        """Calculate overall risk from daily predictions"""
        risk_values = {
            RiskLevel.SAFE: 0,
            RiskLevel.WARNING: 1,
            RiskLevel.DANGER: 2,
            RiskLevel.EXTREME: 3
        }
        max_risk = max(pred['risk_level'] for pred in predictions)
        return max_risk


# Initialize model service
model_service = ModelService()


# API Routes

@app.get("/", response_model=Dict)
async def root():
    """API root endpoint"""
    return {
        "service": "ClimateGuard API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        services={
            "api": "operational",
            "database": "operational",
            "ml_models": "operational",
            "cache": "operational"
        },
        model_version="stgcn-v1.0"
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_risk(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate climate risk predictions for specified regions
    
    - **region_ids**: List of region IDs to predict
    - **disaster_type**: Type of disaster (flood, heatwave, or both)
    - **forecast_days**: Number of days to forecast (1-30)
    """
    # Validate regions
    for region_id in request.region_ids:
        if not RegionDB.get_region(region_id):
            raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
    
    # Generate predictions based on disaster type
    if request.disaster_type == DisasterType.FLOOD:
        predictions = model_service.predict_flood(
            request.region_ids, 
            request.forecast_days
        )
    elif request.disaster_type == DisasterType.HEATWAVE:
        predictions = model_service.predict_heatwave(
            request.region_ids,
            request.forecast_days
        )
    else:  # Both
        flood_preds = model_service.predict_flood(request.region_ids, request.forecast_days)
        heat_preds = model_service.predict_heatwave(request.region_ids, request.forecast_days)
        predictions = flood_preds + heat_preds
    
    return PredictionResponse(
        timestamp=datetime.now(),
        disaster_type=request.disaster_type,
        forecast_days=request.forecast_days,
        regions=predictions,
        metadata={
            "model": "STGCN",
            "data_sources": ["weather_api", "satellite", "ground_stations"],
            "last_updated": datetime.now().isoformat()
        }
    )


@app.get("/api/v1/regions", response_model=List[Dict])
async def get_regions(api_key: str = Depends(verify_api_key)):
    """Get list of all available regions"""
    return RegionDB.get_all_regions()


@app.get("/api/v1/regions/{region_id}", response_model=Dict)
async def get_region(region_id: int, api_key: str = Depends(verify_api_key)):
    """Get details for a specific region"""
    region = RegionDB.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail="Region not found")
    return region


@app.post("/api/v1/historical", response_model=HistoricalDataResponse)
async def get_historical_data(
    request: HistoricalDataRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieve historical climate data for a region
    
    - **region_id**: Region ID
    - **start_date**: Start date for data retrieval
    - **end_date**: End date for data retrieval
    - **variables**: Optional list of climate variables
    """
    region = RegionDB.get_region(request.region_id)
    if not region:
        raise HTTPException(status_code=404, detail="Region not found")
    
    # Mock historical data generation
    days = (request.end_date - request.start_date).days
    data = []
    
    for day in range(days):
        date = request.start_date + timedelta(days=day)
        data.append({
            "date": date.isoformat(),
            "temperature": round(25 + np.random.randn() * 5, 1),
            "rainfall": round(max(0, 15 + np.random.randn() * 20), 1),
            "humidity": round(60 + np.random.randn() * 15, 1),
            "aqi": int(max(0, 50 + np.random.randn() * 30))
        })
    
    statistics = {
        "avg_temperature": round(np.mean([d['temperature'] for d in data]), 2),
        "total_rainfall": round(sum([d['rainfall'] for d in data]), 2),
        "avg_humidity": round(np.mean([d['humidity'] for d in data]), 2),
        "avg_aqi": round(np.mean([d['aqi'] for d in data]), 2)
    }
    
    return HistoricalDataResponse(
        region_id=request.region_id,
        region_name=region['name'],
        start_date=request.start_date,
        end_date=request.end_date,
        data=data,
        statistics=statistics
    )


@app.post("/api/v1/alerts/subscribe")
async def subscribe_alerts(
    subscription: AlertSubscription,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Subscribe to automated climate risk alerts
    
    - **user_id**: Unique user identifier
    - **region_ids**: Regions to monitor
    - **disaster_types**: Types of disasters to monitor
    - **risk_threshold**: Minimum risk level for alerts
    - **notification_channels**: Preferred notification methods
    """
    # Validate regions
    for region_id in subscription.region_ids:
        if not RegionDB.get_region(region_id):
            raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
    
    # In production, save subscription to database
    subscription_id = f"sub-{subscription.user_id}-{datetime.now().timestamp()}"
    
    # Background task to set up monitoring
    background_tasks.add_task(setup_alert_monitoring, subscription)
    
    return {
        "subscription_id": subscription_id,
        "status": "active",
        "message": "Alert subscription created successfully",
        "monitored_regions": len(subscription.region_ids),
        "created_at": datetime.now().isoformat()
    }


@app.get("/api/v1/alerts/active", response_model=List[AlertResponse])
async def get_active_alerts(
    region_id: Optional[int] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get currently active climate risk alerts
    
    - **region_id**: Optional filter by region
    """
    # Mock active alerts
    alerts = [
        AlertResponse(
            alert_id="alert-001",
            timestamp=datetime.now(),
            region_id=1,
            region_name="Central District",
            disaster_type=DisasterType.FLOOD,
            risk_level=RiskLevel.WARNING,
            probability=0.65,
            message="Moderate flood risk expected in the next 48 hours",
            recommendations=[
                "Monitor local weather updates",
                "Prepare emergency supplies",
                "Clear drainage systems"
            ]
        )
    ]
    
    if region_id:
        alerts = [a for a in alerts if a.region_id == region_id]
    
    return alerts


@app.get("/api/v1/stats/summary", response_model=Dict)
async def get_statistics_summary(api_key: str = Depends(verify_api_key)):
    """Get platform statistics summary"""
    return {
        "total_regions": len(RegionDB.regions),
        "active_alerts": 3,
        "predictions_today": 1247,
        "accuracy_rate": 0.87,
        "uptime_hours": 720,
        "last_model_update": (datetime.now() - timedelta(days=2)).isoformat()
    }


# Background tasks
async def setup_alert_monitoring(subscription: AlertSubscription):
    """Background task to set up alert monitoring"""
    # In production, configure monitoring jobs
    print(f"Setting up alert monitoring for user {subscription.user_id}")
    # Could integrate with Celery for scheduled tasks


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("ClimateGuard API starting up...")
    print("Loading ML models...")
    # In production: Initialize database connections, load models, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("ClimateGuard API shutting down...")
    # In production: Close database connections, save state, etc.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)