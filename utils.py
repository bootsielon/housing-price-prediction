"""
utils.py

This module provides utility functions for ranking machine learning models based on various metrics.
It allows for printing top models based on specific metrics and calculating an overall rank across multiple metrics.

Functions:
- print_top_models: Ranks and prints models based on specified metrics.
- calculate_overall_rank: Calculates an overall rank for each model by averaging its ranks across different metrics.
"""
import pandas as pd


def print_top_models(df: pd.DataFrame, metrics: list) -> None:  # df = model evaluation results
    """ Print the top models ranked by the given metrics.
    Args: df (pd.DataFrame): The DataFrame containing model evaluation results.
        metrics (list): A list of metric names by which to rank and print the models. """
    for metric in metrics:  # Iterate over each metric and sort models by the metric
        ascending = True if metric not in ['R2', 'Adjusted R2'] else False  # Determine if sorting should be ascending or descending
        top = df.sort_values(metric, ascending=ascending)  # Rank models by the current metric
        print(f"\nRanked models by {metric}:") # Print ranked models for the metric
        print(top[[metric]])


def calculate_overall_rank(df: pd.DataFrame) -> pd.DataFrame:
    """ Calculate the overall rank of models based on the ranks of individual metrics.
    Args:  df (pd.DataFrame): The DataFrame containing model evaluation results. 
    Returns: pd.DataFrame: The DataFrame with an additional 'Overall_rank' column, sorted by the overall rank. """
    for col in df.columns:  # Rank models by each metric (higher is better for R2 and Adjusted R2, lower is better for others)
        if col in ['R2', 'Adjusted R2']:
            df[f'{col}_rank'] = df[col].rank(ascending=False)
        else:
            df[f'{col}_rank'] = df[col].rank()
    df['Overall_rank'] = df[[col for col in df.columns if col.endswith('_rank')]].mean(axis=1)  # Calculate the overall rank as the mean of individual ranks
    return df.sort_values('Overall_rank')  # Return the DataFrame sorted by the overall rank