# data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt

def print_data_info(data):
    """Print basic information about the dataset."""
    print(f"Data shape: {data.shape}")
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values:")
    print(data.isna().sum())

def plot_histograms(data):
    """Plot histograms for all numerical features."""
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

def plot_categorical(data, column):
    """Plot bar chart for a categorical feature."""
    data[column].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_scatter(data, x, y, alpha=0.1):
    """Plot scatter plot for two features."""
    data.plot(kind='scatter', x=x, y=y, alpha=alpha)
    plt.title(f"Scatter plot: {x} vs {y}")
    plt.show()

def plot_correlation_matrix(data):
    """Plot correlation matrix for numerical features."""
    corr_matrix = data.select_dtypes(include=[float, int]).corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()