# feature_engineering.py

import pandas as pd

def create_features(data):
    """Create new features based on existing ones."""
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data

def encode_categorical(data, columns):
    """Perform one-hot encoding for categorical variables."""
    return pd.get_dummies(data, columns=columns)