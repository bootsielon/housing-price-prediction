"""visualization.py
This module provides functions to visualize model evaluation results. It includes
functions to plot metric comparisons, correlation heatmaps, and scatter plots to
analyze the performance of machine learning models.

Functions:
- save_plot: Saves a plot in the 'plots' directory.
- plot_metric_comparison: Plots a bar chart comparing models based on a specific metric.
- plot_correlation_heatmap: Visualizes the correlation between different evaluation metrics.
- plot_r2_vs_time: Plots a scatter plot of R² score vs computation time for all models. """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for saving plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

def save_plot(fig: plt.Figure, filename: str) -> None:
    """ Save the given figure with the specified filename in the 'plots' directory.
    Args: fig (matplotlib.figure.Figure): The figure to save.
          filename (str): The filename for the plot.  """
    filepath = os.path.join('plots', filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved: {filepath}")


def plot_metric_comparison(df: pd.DataFrame, metric: str, top_n: int = 10) -> None:
    """  Plot a bar chart comparing the top N models based on a specific metric.
    Args:
        df (pd.DataFrame): The DataFrame containing model evaluation results.
        metric (str): The name of the metric to compare.
        top_n (int, optional): The number of top models to display. Defaults to 10.  """
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values(metric, ascending=True)
    sns.barplot(x=df_sorted[metric][:top_n], y=df_sorted.index[:top_n], ax=ax)
    ax.set_title(f'Top {top_n} Models by {metric}')
    ax.set_xlabel(metric)
    plt.tight_layout()
    for i, v in enumerate(df_sorted[metric][:top_n]):  # Add text labels to the bars
        ax.text(v, i, f' {v:.4f}', va='center')
    save_plot(fig, f'top_{top_n}_models_by_{metric.lower()}.png')


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """    Plot a correlation heatmap of the evaluation metrics.
    Args: df (pd.DataFrame): The DataFrame containing model evaluation results. """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation between Metrics')
    plt.tight_layout()
    save_plot(fig, 'correlation_heatmap.png')


def plot_r2_vs_time(df: pd.DataFrame) -> None:
    """ Plot a scatter plot of R2 score vs computation time for all models.
    Args: df (pd.DataFrame): The DataFrame containing model evaluation results."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_col = 'Computation Time (s)'
    r2_col = 'R2'
    
    if time_col not in df.columns:
        print(f"Error: '{time_col}' column not found in the DataFrame.")
        print("Available columns:", df.columns.tolist())
        return
    
    if r2_col not in df.columns:
        print(f"Error: '{r2_col}' column not found in the DataFrame.")
        print("Available columns:", df.columns.tolist())
        return
    
    sns.scatterplot(x=time_col, y=r2_col, data=df, ax=ax)
    
    for i, model in enumerate(df.index):
        ax.annotate(model, (df[time_col][i], df[r2_col][i]))
    
    ax.set_title('R2 Score vs Computation Time')
    plt.tight_layout()
    save_plot(fig, 'r2_score_vs_computation_time.png')