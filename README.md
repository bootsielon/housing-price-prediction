# Housing Price Prediction

This repository contains a complete machine learning pipeline for predicting housing prices in California based on various features such as location, demographics, and housing characteristics. The project includes data loading, exploratory data analysis, feature engineering, model training, evaluation, and deployment of a FastAPI web service for making predictions. It aims to provide accurate price estimates for potential home buyers, sellers, and real estate professionals.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Source and Preprocessing](#data-source-and-preprocessing)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Configuration and Modules](#configuration-and-modules)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Analysis and Model Training](#data-analysis-and-model-training)
  - [Running the API](#running-the-api)
    - [Locally](#locally)
    - [Using Docker](#using-docker)
- [API Documentation](#api-documentation)
  - [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Results](#results)
- [Future Improvements and Known Issues](#future-improvements-and-known-issues)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build and deploy a machine learning model that predicts housing prices in California based on various features such as location, size, and proximity to the ocean. The project utilizes several regression models, compares their performance, and deploys the best-performing model using FastAPI. It aims to provide accurate price estimates for potential home buyers, sellers, and real estate professionals.

## Data Source and Preprocessing

This project uses the **California Housing Prices dataset** from the 1990 Census. The dataset can be obtained from [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices).

### Preprocessing Steps

1. **Handling Missing Values**: Fill missing values with median values.
2. **Feature Scaling**: Use `RobustScaler` to scale numerical features.
3. **Feature Engineering**: Create new features such as `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.
4. **Encoding Categorical Variables**: One-hot encode the `ocean_proximity` categorical feature.

For detailed preprocessing steps, refer to the `data_loader.py` and `feature_engineering.py` modules.

## Features

- **Data Loading and Preprocessing**: Load data from CSV, handle missing values, encode categorical variables, and scale features.
- **Exploratory Data Analysis (EDA)**: Generate plots and statistical summaries to understand the data.
- **Feature Engineering**: Create new features to improve model performance.
- **Model Training and Evaluation**: Train multiple regression models and evaluate them using various metrics.
- **Model Selection**: Compare models and select the best one based on evaluation metrics.
- **API Deployment**: Deploy the model using FastAPI to serve predictions via a web service.
- **Dockerization**: Containerize the API for easy deployment using Docker.

## Repository Structure

```
housing-price-prediction/
├── api.py
├── config.py
├── data_analysis.py
├── data_loader.py
├── Dockerfile
├── feature_engineering.py
├── main.py
├── model_training.py
├── requirements.txt
├── utils.py
├── visualization.py
├── .dockerignore
├── .gitignore
├── README.md
├── plots/
│   ├── histograms.png
│   ├── categorical_ocean_proximity.png
│   ├── scatter_longitude_vs_latitude.png
│   ├── correlation_matrix.png
│   ├── top_10_models_by_MSE.png
│   ├── top_10_models_by_R2.png
│   ├── correlation_heatmap.png
│   └── r2_score_vs_computation_time.png
├── best_model.joblib
├── robust_scaler.joblib
├── version_info.txt
└── data/
    └── housing.csv
```

## Configuration and Modules

### Configuration (`config.py`)

Contains configuration settings for the project, including:

- `DATA_PATH`: Path to the housing dataset CSV file (`data/housing.csv`).
- `RANDOM_STATE`: The seed for random number generation to ensure reproducibility (`42`).
- `TEST_SIZE`: The proportion of the dataset to include in the test split (`0.2`).

### Modules

#### `data_loader.py`

- **Functions**:
  - `load_data()`: Loads housing data from a CSV file.
  - `split_data(data)`: Splits the data into training and test sets using stratified sampling.
  - `preprocess_data(train_set, test_set)`: Preprocesses the training and test sets by encoding, handling missing values, and scaling features.

#### `data_analysis.py`

- **Functions**:
  - `print_data_info(data)`: Prints basic dataset information such as shape, data types, and missing values.
  - `plot_histograms(data)`: Plots histograms for all numerical features in the dataset.
  - `plot_categorical(data, column)`: Plots a bar chart for a specified categorical feature.
  - `plot_scatter(data, x, y)`: Plots a scatter plot for two features to visualize their relationship.
  - `plot_correlation_matrix(data)`: Generates a heatmap of the correlation matrix for numerical features.

#### `feature_engineering.py`

- **Functions**:
  - `create_features(data)`: Generates new features such as `rooms_per_household` and `bedrooms_per_room`.
  - `encode_categorical(data, columns)`: Performs one-hot encoding on specified categorical columns.

#### `model_training.py`

- **Functions**:
  - `train_predict_evaluate(model, X_train, y_train, X_test, y_test)`: Trains a given model, makes predictions, and computes evaluation metrics.
  - `train_and_evaluate_models(X_train, y_train, X_test, y_test)`: Trains and evaluates a list of models and returns a comparison of results and the best model.

#### `visualization.py`

- **Functions**:
  - `plot_metric_comparison(df, metric)`: Plots a bar chart comparing models based on a specific metric.
  - `plot_correlation_heatmap(df)`: Visualizes the correlation between different evaluation metrics.
  - `plot_r2_vs_time(df)`: Plots a scatter plot of R² score vs computation time for all models.

#### `utils.py`

- **Functions**:
  - `print_top_models(df, metrics)`: Ranks and prints models based on specified metrics.
  - `calculate_overall_rank(df)`: Calculates an overall rank for each model by averaging its ranks across different metrics.

#### `api.py`

Implements a FastAPI-based web service for housing price prediction using the pre-trained machine learning model.

- **Endpoints**:
  - `POST /predict`: Accepts housing features and returns the predicted house price.
  - `GET /`: A simple root endpoint for API health check.

#### `main.py`

Executes the end-to-end machine learning pipeline:

1. Load data and split into training and test sets.
2. Perform exploratory data analysis with visualizations.
3. Engineer new features.
4. Preprocess the data by encoding and scaling.
5. Train multiple models and evaluate their performance.
6. Save the trained models and relevant artifacts.
7. Visualize the model comparison results and rank the models.

## Installation

### Prerequisites

- **Python 3.8 or higher** (Developed and tested with Python 3.12)
- **Git**
- **Docker** (optional, for containerization)
- **pip** (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Data Analysis and Model Training

To run the entire machine learning pipeline, including data loading, preprocessing, feature engineering, model training, and evaluation, execute the `main.py` script:

```bash
python main.py
```

This script will:

- Load and preprocess the data.
- Perform exploratory data analysis and save plots to the `plots/` directory.
- Engineer new features.
- Train multiple models and evaluate their performance.
- Save the best-performing model and scaler.
- Generate visualizations comparing model performance.

### Running the API

Ensure that `best_model.joblib` and `robust_scaler.joblib` are present in the project root after running `main.py`.

#### Locally

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

#### Using Docker

Build the Docker image:

```bash
docker build -t housing-api .
```

Run the Docker container:

```bash
docker run -p 8000:8000 housing-api
```

## API Documentation

Once the server is running, you can access the API documentation at `http://localhost:8000/docs`.

### API Endpoints

#### `POST /predict`

Predict the house price based on provided features.

- **Request Body**:

  ```json
  {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }
  ```

- **Response**:

  ```json
  {
    "predicted_price": 452600.0
  }
  ```

#### `GET /`

Health check endpoint to verify that the API is running.

- **Response**:

  ```json
  {
    "message": "Welcome to the Housing Price Prediction API"
  }
  ```

## Testing

To test the API locally:

### Using `curl`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "longitude": -122.23,
           "latitude": 37.88,
           "housing_median_age": 41.0,
           "total_rooms": 880.0,
           "total_bedrooms": 129.0,
           "population": 322.0,
           "households": 126.0,
           "median_income": 8.3252,
           "ocean_proximity": "NEAR BAY"
         }'
```

### Using Python `requests`

Create a file named `test_api.py`:

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}

response = requests.post(url, json=data)
print(response.json())
```

Run the script:

```bash
python test_api.py
```

### Using Swagger UI

Open a web browser and navigate to `http://localhost:8000/docs` to access the interactive API documentation.

## Results

After running the `main.py` script, evaluation metrics for all trained models will be saved to `model_comparison_results.csv`. Plots comparing model performance are saved in the `plots/` directory.

### Top Models

An example of the top models based on overall rank:

| Model            | RMSE    | R2     | Adjusted R2 | MAPE   | MAE    | Overall_rank |
|------------------|---------|--------|-------------|--------|--------|--------------|
| CatBoost         | 50000.0 | 0.85   | 0.85        | 15.0%  | 40000.0| 1            |
| XGBoost          | 51000.0 | 0.84   | 0.84        | 16.0%  | 41000.0| 2            |
| Random Forest    | 52000.0 | 0.83   | 0.83        | 17.0%  | 42000.0| 3            |

*Note: The actual results may vary.*

## Future Improvements and Known Issues

### Future Improvements

1. **Model Updates**: Implement periodic retraining of the model with new data to maintain prediction accuracy over time.
2. **Feature Importance**: Add functionality to explain predictions and show feature importance.
3. **API Authentication**: Implement user authentication for the API to control access and usage.
4. **Caching**: Add caching mechanisms to improve API response times for frequent queries.
5. **Logging and Monitoring**: Enhance logging and add monitoring tools for better observability in production.
6. **Hyperparameter Optimization**: Automate hyperparameter optimization with feature selection to improve model performance.

### Known Issues

- The model is trained on historical data and may not account for recent market trends.
- Predictions may be less accurate for areas with limited data in the training set.
- The current implementation does not handle extreme outliers well.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.

---

*Disclaimer: This project is for educational purposes.*

---