import pandas as pd
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# 1. Setup & Configuration
app = FastAPI(
    title="Agri-Yield 3D Brain",
    description="Enterprise API for Crop Yield Forecasting using Random Forest Multi-Crop Engine",
    version="1.0.0"
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model (Global State)
MODEL_PATH = "models/crop_yield_model.pkl"
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise RuntimeError("Model could not be loaded. Check file path.")

# 2. Define Input Schema (The Contract)
class CropInput(BaseModel):
    # Crop Selection (Must match training data exactly)
    item: Literal['Wheat', 'Rice', 'Maize (corn)', 'Sugar cane', 'Seed cotton, unginned'] = Field(..., description="Target Crop")
    
    # Historical Context
    yield_lag1: float = Field(..., gt=0, description="Previous Year's Yield (hg/ha)")
    
    # Weather - Quarter 1 (Jan-Mar)
    prectotcorr_q1: float = Field(..., ge=0, description="Q1 Rainfall (mm)")
    t2m_max_q1: float = Field(..., description="Q1 Max Temp (C)")
    t2m_min_q1: float = Field(..., description="Q1 Min Temp (C)")
    rh2m_q1: float = Field(..., ge=0, le=100, description="Q1 Relative Humidity (%)")
    
    # Weather - Quarter 2 (Apr-Jun)
    prectotcorr_q2: float = Field(..., ge=0)
    t2m_max_q2: float = Field(...)
    t2m_min_q2: float = Field(...)
    rh2m_q2: float = Field(..., ge=0, le=100)
    
    # Weather - Quarter 3 (Jul-Sep)
    prectotcorr_q3: float = Field(..., ge=0)
    t2m_max_q3: float = Field(...)
    t2m_min_q3: float = Field(...)
    rh2m_q3: float = Field(..., ge=0, le=100)

    # Weather - Quarter 4 (Oct-Dec)
    prectotcorr_q4: float = Field(..., ge=0)
    t2m_max_q4: float = Field(...)
    t2m_min_q4: float = Field(...)
    rh2m_q4: float = Field(..., ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "item": "Wheat",
                "yield_lag1": 25000.0,
                "prectotcorr_q1": 45.5, "t2m_max_q1": 26.5, "t2m_min_q1": 10.2, "rh2m_q1": 55.0,
                "prectotcorr_q2": 12.0, "t2m_max_q2": 38.0, "t2m_min_q2": 22.0, "rh2m_q2": 30.0,
                "prectotcorr_q3": 150.0, "t2m_max_q3": 35.0, "t2m_min_q3": 25.0, "rh2m_q3": 65.0,
                "prectotcorr_q4": 10.0, "t2m_max_q4": 22.0, "t2m_min_q4": 8.0, "rh2m_q4": 45.0
            }
        }

# 3. Prediction Endpoint
@app.post("/predict", tags=["Inference"])
async def predict_yield(data: CropInput):
    """
    Predicts crop yield (hg/ha) based on weather and historical data.
    """
    try:
        # Convert Pydantic object to Pandas DataFrame
        # The key names must EXACTLY match the columns used in 'src/train.py'
        input_data = {
            'Item': [data.item],
            'Yield_Lag1': [data.yield_lag1],
            
            # Q1
            'PRECTOTCORR_Q1': [data.prectotcorr_q1],
            'T2M_MAX_Q1': [data.t2m_max_q1],
            'T2M_MIN_Q1': [data.t2m_min_q1],
            'RH2M_Q1': [data.rh2m_q1],
            
            # Q2
            'PRECTOTCORR_Q2': [data.prectotcorr_q2],
            'T2M_MAX_Q2': [data.t2m_max_q2],
            'T2M_MIN_Q2': [data.t2m_min_q2],
            'RH2M_Q2': [data.rh2m_q2],
            
            # Q3
            'PRECTOTCORR_Q3': [data.prectotcorr_q3],
            'T2M_MAX_Q3': [data.t2m_max_q3],
            'T2M_MIN_Q3': [data.t2m_min_q3],
            'RH2M_Q3': [data.rh2m_q3],
            
            # Q4
            'PRECTOTCORR_Q4': [data.prectotcorr_q4],
            'T2M_MAX_Q4': [data.t2m_max_q4],
            'T2M_MIN_Q4': [data.t2m_min_q4],
            'RH2M_Q4': [data.rh2m_q4],
        }
        
        df = pd.DataFrame(input_data)
        
        # Make Prediction
        prediction = model_pipeline.predict(df)
        predicted_value = float(prediction[0])
        
        # Response Logic
        return {
            "status": "success",
            "crop": data.item,
            "predicted_yield_hg_ha": round(predicted_value, 2),
            "predicted_yield_tonnes_ha": round(predicted_value / 10000, 2),
            "yield_change_percent": round(((predicted_value - data.yield_lag1) / data.yield_lag1) * 100, 2)
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Processing Error")

# 4. Health Check
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "active", "model_version": "v1.0.0_RandomForest"}
