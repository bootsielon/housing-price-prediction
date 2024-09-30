"""
data_analysis.py

This module contains functions for exploratory data analysis (EDA),
including generating plots and displaying basic information about the dataset.
It focuses on visualizing numerical and categorical data, such as histograms,
scatter plots for feature relationships, and correlation matrices.

Functions:
- print_data_info: Prints basic dataset information such as shape, data types, and missing values.
- plot_histograms: Plots histograms for all numerical features in the dataset.
- plot_categorical: Plots a bar chart for a specified categorical feature.
- plot_scatter: Plots a scatter plot for two features to visualize their relationship.
- plot_correlation_matrix: Generates a heatmap of the correlation matrix for numerical features.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.figure import Figure

# Create a 'plots' directory in the current working directory
current_dir = os.getcwd()
plots_dir = os.path.join(current_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

def save_plot(fig: Figure, filename: str) -> None:
    """Save a matplotlib figure to the 'plots' directory and close the plot.
    Args: fig (matplotlib.figure.Figure): The figure object to save.
          filename (str): The name of the file to save the figure as. Should include file extension."""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved: {filepath}")


def print_data_info(data: pd.DataFrame) -> None:
    """ Print basic information about the dataset, including shape, data types, and missing values.
    Args: data (pd.DataFrame): The dataset for which to print information."""
    print(f"Data shape: {data.shape}")
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values:")
    print(data.isna().sum())


def plot_histograms(data: pd.DataFrame) -> None:
    """    Plot histograms for all numerical features in the dataset.
    Args: data (pd.DataFrame): The dataset containing numerical features."""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    axes = axes.flatten()
    for i, column in enumerate(data.select_dtypes(include=['int64', 'float64']).columns):
        if i < 9:  # Limit to 9 subplots
            data[column].hist(ax=axes[i], bins=50)
            axes[i].set_title(column)
    plt.tight_layout()
    save_plot(fig, 'histograms.png')


def plot_categorical(data: pd.DataFrame, column: str) -> None:
    """ Plot a bar chart for a specified categorical feature.
    Args: data (pd.DataFrame): The dataset containing the categorical feature.
          column (str): The name of the categorical feature to plot. """
    fig, ax = plt.subplots(figsize=(10, 6))
    data[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Distribution of {column}")
    plt.tight_layout()
    save_plot(fig, f'categorical_{column}.png')


def plot_scatter(data: pd.DataFrame, x: str, y: str, alpha: float = 0.1) -> None:
    """ Plot a scatter plot to visualize the relationship between two features.
    Args: data (pd.DataFrame): The dataset containing the features.
        x (str): The name of the feature to plot on the x-axis.
        y (str): The name of the feature to plot on the y-axis.
        alpha (float, optional): The transparency level of the points in the scatter plot. Defaults to 0.1. """
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind='scatter', x=x, y=y, alpha=alpha, ax=ax)
    ax.set_title(f"Scatter plot: {x} vs {y}")
    plt.tight_layout()
    save_plot(fig, f'scatter_{x}_vs_{y}.png')


def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """ Plot a heatmap of the correlation matrix for numerical features.
    Args: data (pd.DataFrame): The dataset containing numerical features."""
    corr_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    save_plot(fig, 'correlation_matrix.png')