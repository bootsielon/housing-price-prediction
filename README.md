# housing-price-prediction repo: Housing Price Prediction API

This project implements a machine learning regression model to predict housing prices in California based on various features such as location, size, and proximity to the ocean. It aims to provide accurate price estimates for potential home buyers, sellers, and real estate professionals. The project encompasses data processing, exploratory data analysis, model training, and a REST API for serving predictions, making it a comprehensive tool for real estate market analysis.

## Table of Contents
0. [Data Source and Preprocessing](#data-source-preprocessing)
1. [Project Structure](#project-structure)
2. [Configuration and Module Descriptions](#config-module-descriptions)
3. [Installation and Requirements](#installation-requirements)
4. [Usage](#usage)
5. [API Documentation](#api-documentation)
6. [Docker Deployment](#docker-deployment)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)


## Data Source and Preprocessing

This project uses the California Housing Prices dataset from the 1990 Census. The dataset can be obtained from [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices).

Preprocessing steps include:
1. Handling missing values
2. Feature scaling using RobustScaler
3. Creating new features such as rooms_per_household, bedrooms_per_room, and population_per_household
4. One-hot encoding of categorical variables (ocean_proximity)

For detailed preprocessing steps, refer to the `data_loader.py` and `feature_engineering.py` modules.


## Project Structure
project_directory/
│
├── api.py
├── main.py
├── data_loader.py
├── data_analysis.py
├── feature_engineering.py
├── model_training.py
├── visualization.py
├── utils.py
├── config.py
├── best_model.joblib
├── robust_scaler.joblib
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
└── version_info.txt

## Configuration and Module Descriptions

The project uses the following environment variables:
- `PORT`: The port on which the API will run (default: 8000)

To set these variables, you can create a `.env` file in the project root or set them in your environment before running the application.

Additional configuration settings can be found in the `config.py` file, including:
- `DATA_PATH` = 'data/housing.csv', the data used to train models
- `RANDOM_STATE` = 42
- `TEST_SIZE` = The proportion of data to use for testing (currently 0.2)

### config.py
Configuration settings for the project, including file paths, random seed, and test set size.

### data_loader.py
Handles data loading and preprocessing. Key functions:
- `load_data() -> pd.DataFrame`: Load the housing data from CSV file.
- `split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`: Split data into training and test sets.
- `preprocess_data(train_set: pd.DataFrame, test_set: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: Preprocess the data.

### data_analysis.py
Provides exploratory data analysis functions. Key functions:
- `print_data_info(data: pd.DataFrame) -> None`: Print basic information about the dataset.
- `plot_histograms(data: pd.DataFrame) -> None`: Plot histograms for numerical features.
- `plot_correlation_matrix(data: pd.DataFrame) -> None`: Plot correlation matrix for numerical features.

### feature_engineering.py
Handles feature creation and encoding. Key functions:
- `create_features(data: pd.DataFrame) -> pd.DataFrame`: Create new features based on existing ones.
- `encode_categorical(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame`: Perform one-hot encoding for categorical variables.

### model_training.py
Manages model training and evaluation. Key functions:
- `train_predict_evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float, float, float]`: Train, predict, and evaluate a model.
- `train_and_evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame`: Train and evaluate multiple models.

### visualization.py
Provides data visualization functions. Key functions:
- `plot_metric_comparison(df: pd.DataFrame, metric: str, top_n: int = 10) -> None`: Plot comparison of top models for a specific metric.
- `plot_r2_vs_time(df: pd.DataFrame) -> None`: Plot R2 score vs computation time for all models.

### utils.py
Contains utility functions. Key functions:
- `print_top_models(df: pd.DataFrame, metrics: List[str], top_n: int = 3) -> None`: Print top models for each specified metric.
- `calculate_overall_rank(df: pd.DataFrame) -> pd.DataFrame`: Calculate overall rank of models based on all metrics.

### main.py
The main script that orchestrates the entire data analysis and model evaluation pipeline.

## Installation and Requirements

### Requirements

Before installing and running this project, ensure you have the following prerequisites:

- Python 3.12
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- Git (for cloning the repository)

Additionally, make sure you have sufficient disk space for the dataset and model files (approximately 500MB recommended).

For optimal performance, a system with at least 4GB of RAM and a multi-core processor is recommended, especially when training models or handling multiple API requests.

Note: This project is developed and tested with Python 3.12. While it may work with other Python 3.x versions, Python 3.12 is recommended for the best compatibility and performance.

### Installation

1. Clone the repository:
   
   git clone https://github.com/bootsielon/housing-price-prediction.git
   cd housing-price-prediction
   

2. Create a virtual environment (optional but recommended):
   
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   

3. Install the required packages:
   
   pip install -r requirements.txt


## Usage

1. To train the model and perform data analysis:
   
   python main.py
   

2. To start the API server locally:
   
   uvicorn api:app --reload
   

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the API documentation at `http://localhost:8000/docs`.

### Predict Housing Price

Endpoint: `POST /predict`

Request Body:
```json
{
  "longitude": float,
  "latitude": float,
  "housing_median_age": float,
  "total_rooms": float,
  "total_bedrooms": float,
  "population": float,
  "households": float,
  "median_income": float,
  "ocean_proximity": string
}
```

Response:
```json
{
  "predicted_price": float
}
```

## Docker Deployment

1. Build the Docker image:
   
   docker build -t housing-price-prediction-api .
   

2. Run the Docker container:
   
   docker run -p 8000:8000 housing-price-prediction-api
   

The API will be available at `http://localhost:8000`.

## Testing

To test the API locally:

1. Using curl:
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
              "ocean_proximity": "<1H OCEAN"
            }'
   ```

2. Using Python requests:
   Create a file named `test_api.py` with the following content:

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
       "ocean_proximity": "<1H OCEAN"
   }

   response = requests.post(url, json=data)
   print(response.json())
   ```

   Run the script:
   
   python test_api.py
   

3. Using the Swagger UI:
   Open a web browser and go to `http://localhost:8000/docs`.


## Future Improvements and Known Issues

While this project provides a functional housing price prediction model and API, there are several areas for potential improvement:

1. **Model updates**: Implement periodic retraining of the model with new data to maintain prediction accuracy over time.
2. **Feature importance**: Add functionality to explain predictions and show feature importance.
3. **API authentication**: Implement user authentication for the API to control access and usage.
4. **Caching**: Add caching mechanisms to improve API response times for frequent queries.
5. **Logging and monitoring**: Enhance logging and add monitoring tools for better observability in production.
6. **Hyperparameter Optimization**: Automate hyperparameter optimization with feature selection from feature importance to compare the best possible models from all model families.

Known limitations:
- The model is trained on historical data and may not account for recent market trends.
- Predictions may be less accurate for areas with limited data in the training set.
- The current implementation does not handle extreme outliers well.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.