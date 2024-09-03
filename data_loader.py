# data_loader.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE
from typing import Tuple

def load_data():
    """Load the housing data from CSV file."""
    return pd.read_csv(DATA_PATH)

def split_data(data):
    """Split data into training and test sets using stratified sampling."""
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in split.split(data, data['ocean_proximity']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set

def preprocess_data(
        train_set: pd.DataFrame, test_set: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Preprocess the data: encode categorical variables, handle missing values, and scale features.
    
    Args:
        train_set (pd.DataFrame): The training set.
        test_set (pd.DataFrame): The test set.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]: 
        Preprocessed X_train, y_train, X_test, y_test, and the fitted scaler.
    """    
    train_set_encoded = pd.get_dummies(train_set, columns=['ocean_proximity'])
    test_set_encoded = pd.get_dummies(test_set, columns=['ocean_proximity'])
    
    median_values = train_set_encoded.median()
    train_set_encoded.fillna(median_values, inplace=True)
    test_set_encoded.fillna(median_values, inplace=True)
    
    X_train = train_set_encoded.drop(columns='median_house_value')
    y_train = train_set_encoded['median_house_value']
    X_test = test_set_encoded.drop(columns='median_house_value')
    y_test = test_set_encoded['median_house_value']
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler