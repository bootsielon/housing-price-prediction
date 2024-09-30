"""
data_loader.py

This module provides functionality to load, split, and preprocess housing data.
The data is loaded from a CSV file, split into training and test sets using stratified sampling,
and then preprocessed by encoding categorical variables, handling missing values, and scaling features.

Functions:
- load_data: Loads housing data from a CSV file.
- split_data: Splits the data into training and test sets based on the 'ocean_proximity' feature.
- preprocess_data: Preprocesses the training and test sets by encoding, handling missing values, and scaling features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE
from typing import Tuple


def load_data() -> pd.DataFrame:
    """Load the housing data from CSV file. Returns: pd.DataFrame: The housing dataset."""
    return pd.read_csv(DATA_PATH)


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split the data into training and test sets using stratified sampling
    based on the 'ocean_proximity' feature.
    Args: data (pd.DataFrame): The dataset to be split.
    Returns: Tuple[pd.DataFrame, pd.DataFrame]: The stratified training and test sets. """
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in split.split(data, data['ocean_proximity']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set


def preprocess_data(
        train_set: pd.DataFrame, test_set: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """ Preprocess the data by encoding categorical variables, handling missing values,
    and scaling the features using RobustScaler.

    Args:
        train_set (pd.DataFrame): The training set.
        test_set (pd.DataFrame): The test set.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
        - X_train (np.ndarray): Scaled features for the training set.
        - y_train (np.ndarray): Target variable for the training set.
        - X_test (np.ndarray): Scaled features for the test set.
        - y_test (np.ndarray): Target variable for the test set.
        - scaler (RobustScaler): Fitted scaler for future use.    """    
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