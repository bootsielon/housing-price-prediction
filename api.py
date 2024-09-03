# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Load the scaler
scaler = joblib.load('robust_scaler.joblib')

app = FastAPI()

class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.post("/predict")
async def predict_house_price(features: HousingFeatures):
    try:
        # Convert input features to numpy array
        feature_array = np.array([[
            features.longitude,
            features.latitude,
            features.housing_median_age,
            features.total_rooms,
            features.total_bedrooms,
            features.population,
            features.households,
            features.median_income
        ]])

        # One-hot encode the ocean_proximity
        ocean_proximity_encoding = {
            '<1H OCEAN': [1, 0, 0, 0, 0],
            'INLAND': [0, 1, 0, 0, 0],
            'ISLAND': [0, 0, 1, 0, 0],
            'NEAR BAY': [0, 0, 0, 1, 0],
            'NEAR OCEAN': [0, 0, 0, 0, 1]
        }

        if features.ocean_proximity not in ocean_proximity_encoding:
            raise HTTPException(status_code=400, detail="Invalid ocean_proximity value")

        encoded_ocean_proximity = ocean_proximity_encoding[features.ocean_proximity]
        feature_array = np.concatenate([feature_array, np.array([encoded_ocean_proximity])], axis=1)

        # Scale the features
        scaled_features = scaler.transform(feature_array)

        # Make prediction
        prediction = model.predict(scaled_features)

        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Housing Price Prediction API"}