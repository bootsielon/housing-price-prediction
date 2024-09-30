"""
api.py

This module implements a FastAPI-based web service for housing price prediction using a pre-trained machine learning model.
The API accepts housing features as input, applies feature engineering and scaling, and returns a predicted house price.

Endpoints:
- POST /predict: Accepts housing features and returns the predicted house price.
- GET /: A simple root endpoint for API health check.

Functions:
- load_file: Tries to load a model or scaler using joblib, and falls back to pickle if joblib fails.
- predict_house_price: The main prediction logic for processing input features and returning the predicted price.
"""

from feature_engineering import create_features
import sys
from sklearn import __version__ as sklearn_version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

print(f"API running with Python version: {sys.version}")
print(f"API running with scikit-learn version: {sklearn_version}")


def load_file(joblib_path: str, pickle_path: str):
    """
    Load a model or scaler from a file using joblib, with a fallback to pickle if joblib fails.

    Args:
        joblib_path (str): The path to the joblib file.
        pickle_path (str): The path to the pickle file.

    Returns:
        The loaded model or scaler, or None if both loading methods fail.
    """
    try:
        return joblib.load(joblib_path)
    except Exception as e:
        print(f"Error loading with joblib: {e}")
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading with pickle: {e}")
            return None
# Load model and scaler, and check version compatibility
try:
    # Check version compatibility
    with open('version_info.txt', 'r') as f:
        version_info = f.read()
    print("Model was trained with:")
    print(version_info)
    
    if sklearn_version != version_info.split('\n')[1].split(': ')[1]:
        print("Warning: Current scikit-learn version differs from the version used for training.")

    current_dir = os.getcwd()
    model_joblib_path = os.path.join(current_dir, 'best_model.joblib')
    model_pickle_path = os.path.join(current_dir, 'best_model.pkl')
    scaler_joblib_path = os.path.join(current_dir, 'robust_scaler.joblib')
    scaler_pickle_path = os.path.join(current_dir, 'robust_scaler.pkl')
    
    print(f"Attempting to load model...")
    model = load_file(model_joblib_path, model_pickle_path)
    
    print(f"Attempting to load scaler...")
    scaler = load_file(scaler_joblib_path, scaler_pickle_path)
    
    if model is not None and scaler is not None:
        print("Model and scaler loaded successfully")
    else:
        print("Failed to load model or scaler")
except Exception as e:
    print(f"Error during loading process: {e}")
    model = None
    scaler = None


class HousingFeatures(BaseModel):  # Define the input data structure
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
    """ 
    Predict the house price based on the provided housing features.

    Args:
        features (HousingFeatures): The input housing features.

    Returns:
        dict: The predicted house price.
    """

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Please check server logs.")
    try:
        # Convert input features to a pandas DataFrame
        feature_df = pd.DataFrame({
            'longitude': [features.longitude],
            'latitude': [features.latitude],
            'housing_median_age': [features.housing_median_age],
            'total_rooms': [features.total_rooms],
            'total_bedrooms': [features.total_bedrooms],
            'population': [features.population],
            'households': [features.households],
            'median_income': [features.median_income],
            'ocean_proximity': [features.ocean_proximity]
        })
        # Apply feature engineering
        feature_df = create_features(feature_df)
        # One-hot encode the ocean_proximity
        feature_df = pd.get_dummies(feature_df, columns=['ocean_proximity'])
        # Ensure all expected columns are present
        expected_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                            'total_bedrooms', 'population', 'households', 'median_income',
                            'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
                            'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
                            'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
                            'ocean_proximity_NEAR OCEAN']
        for col in expected_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        # Reorder columns to match the order expected by the scaler
        feature_df = feature_df[expected_columns]
        # Convert to numpy array
        feature_array = feature_df.values
        print(f"Feature array shape before scaling: {feature_array.shape}")
        print(f"Feature array content: {feature_array}")
        # Scale the features
        scaled_features = scaler.transform(feature_array)
        print(f"Scaled features shape: {scaled_features.shape}")
        # Make prediction
        prediction = model.predict(scaled_features)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    

@app.get("/")
async def root():
    """ Root endpoint to verify that the API is running. Returns A welcome message in dict form. """    
    return {"message": "Welcome to the Housing Price Prediction API"}