"""
model_training.py

This module contains functions for training, evaluating, and comparing machine learning models.
It supports various regression models and computes multiple evaluation metrics, including R², RMSE, MAE, and adjusted R².
The module can train and evaluate multiple models and returns the best-performing model based on the R² score.

Functions:
- adjusted_r2: Computes the adjusted R² score to account for model complexity.
- mean_absolute_percentage_error: Computes the MAPE, a percentage-based error metric.
- train_predict_evaluate: Trains a given model, makes predictions, and computes evaluation metrics.
- train_and_evaluate_models: Trains and evaluates a list of models and returns a comparison of results and the best model.
"""


import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from typing import Any, Tuple


def adjusted_r2(r2: float, n: int, p: int) -> float:  # n = number of samples, p = number of features
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # MAPE
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_predict_evaluate(
        model: Any,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[pd.DataFrame, Any]:
    """Train the model, make predictions, and evaluate performance on test data.
    Args: model (Any): The machine learning model to train.
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test target values.
    Returns: Tuple[float, float, float, float, float, float, float]:
            - MSE (float): Mean Squared Error.
            - RMSE (float): Root Mean Squared Error.
            - MAE (float): Mean Absolute Error.
            - R² (float): R² score.
            - Adjusted R² (float): Adjusted R² score.
            - MAPE (float): Mean Absolute Percentage Error.
            - Computation Time (float): Time taken to train and predict in seconds."""
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])
    mape = mean_absolute_percentage_error(y_test, y_pred)
    computation_time = end_time - start_time
    return mse, rmse, mae, r2, adj_r2, mape, computation_time


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """ Train multiple regression models, evaluate their performance, and return the results
    along with the best-performing model.
    Args: X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test target values.
    Returns:
        Tuple[pd.DataFrame, Any]:
            - pd.DataFrame: A DataFrame with the evaluation results of all models.
            - Any: The best-performing model based on the R² score."""
    models = [
        ('Linear Regression', LinearRegression()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('SVR', SVR()),
        # ('MLP', MLPRegressor()),
        ('Gradient Boosting', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor()),
        ('LightGBM', LGBMRegressor()),
        ('CatBoost', CatBoostRegressor(verbose=0)),
        ('Ridge', Ridge()),
        ('Lasso', Lasso()),
        ('ElasticNet', ElasticNet()),
    ]
    results = []
    best_model = None
    best_r2 = -float('inf')
    for name, model in models: # Iterate through each model family, train and evaluate it
        mse, rmse, mae, r2, adj_r2, mape, comp_time = train_predict_evaluate(model, X_train, y_train, X_test, y_test)
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Adjusted R2': adj_r2,
            'MAPE': mape,
            'Computation Time (s)': comp_time
        })
        # Keep track of the best-performing model based on R² score
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    return pd.DataFrame(results).set_index('Model').round(4), best_model