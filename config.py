"""config.py

This configuration file defines constants used across the machine learning pipeline, 
such as the path to the dataset, the random seed for reproducibility, and the test set size.

Constants:
- DATA_PATH: Path to the housing dataset CSV file.
- RANDOM_STATE: The seed for random number generation to ensure reproducibility.
- TEST_SIZE: The proportion of the dataset to include in the test split. """


# Path to the dataset
DATA_PATH = 'data/housing.csv'

# Random seed for reproducibility
RANDOM_STATE = 42

# Proportion of the dataset to include in the test split
TEST_SIZE = 0.2
