"""feature_engineering.py

This module provides functions to create new features and encode categorical variables
for a dataset. Feature engineering helps improve model performance by adding meaningful 
derived variables, and encoding transforms categorical features into a format suitable 
for machine learning models.

Functions:
- create_features: Generates new features such as 'rooms_per_household' and 'bedrooms_per_room'.
- encode_categorical: Performs one-hot encoding on specified categorical columns.
"""
import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create new features based on existing ones.
    Args: data (pd.DataFrame): The dataset containing the original features.
    Returns: pd.DataFrame: The dataset with new derived features."""
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data

def encode_categorical(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Perform one-hot encoding for categorical variables."""
    return pd.get_dummies(data, columns=columns)