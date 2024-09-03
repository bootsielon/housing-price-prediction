# model_training.py

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


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_predict_evaluate(
        model: Any,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[pd.DataFrame, Any]:
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

    for name, model in models:
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

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return pd.DataFrame(results).set_index('Model').round(4), best_model