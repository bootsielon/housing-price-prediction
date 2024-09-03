# data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a 'plots' directory in the current working directory
current_dir = os.getcwd()
plots_dir = os.path.join(current_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

def save_plot(fig, filename):
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved: {filepath}")

def print_data_info(data):
    """Print basic information about the dataset."""
    print(f"Data shape: {data.shape}")
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values:")
    print(data.isna().sum())

def plot_histograms(data):
    """Plot histograms for all numerical features."""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, column in enumerate(data.select_dtypes(include=['int64', 'float64']).columns):
        if i < 9:  # Limit to 9 subplots
            data[column].hist(ax=axes[i], bins=50)
            axes[i].set_title(column)
    
    plt.tight_layout()
    save_plot(fig, 'histograms.png')

def plot_categorical(data, column):
    """Plot bar chart for a categorical feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Distribution of {column}")
    plt.tight_layout()
    save_plot(fig, f'categorical_{column}.png')

def plot_scatter(data, x, y, alpha=0.1):
    """Plot scatter plot for two features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind='scatter', x=x, y=y, alpha=alpha, ax=ax)
    ax.set_title(f"Scatter plot: {x} vs {y}")
    plt.tight_layout()
    save_plot(fig, f'scatter_{x}_vs_{y}.png')

def plot_correlation_matrix(data):
    """Plot correlation matrix for numerical features."""
    corr_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    save_plot(fig, 'correlation_matrix.png')