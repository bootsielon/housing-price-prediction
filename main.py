# main.py

import pandas as pd
from data_loader import load_data, split_data, preprocess_data
from data_analysis import print_data_info, plot_histograms, plot_categorical, plot_scatter, plot_correlation_matrix
from feature_engineering import create_features, encode_categorical
from model_training import train_and_evaluate_models
from visualization import plot_metric_comparison, plot_correlation_heatmap, plot_r2_vs_time
from utils import print_top_models, calculate_overall_rank

def main():
    # Load and split data
    data = load_data()
    train_set, test_set = split_data(data)
    
    # Perform data analysis
    print_data_info(data)
    plot_histograms(data)
    plot_categorical(data, 'ocean_proximity')
    plot_scatter(data, 'longitude', 'latitude')
    plot_correlation_matrix(data)
    
    # Feature engineering
    train_set = create_features(train_set)
    test_set = create_features(test_set)
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_set, test_set)
    
    # Train and evaluate models
    results_df = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Visualize results
    plot_metric_comparison(results_df, 'MSE')
    plot_metric_comparison(results_df, 'R2')
    plot_correlation_heatmap(results_df)
    plot_r2_vs_time(results_df)
    
    # Print top models
    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'Adjusted R2', 'MAPE', 'Computation Time (s)']
    print_top_models(results_df, metrics)
    
    # Calculate overall ranking
    ranked_df = calculate_overall_rank(results_df)
    print("\nTop 5 models overall:")
    print(ranked_df[['MSE', 'R2', 'MAPE', 'Computation Time (s)', 'Overall_rank']].head())
    
    # Save results
    results_df.to_csv('model_comparison_results.csv')

if __name__ == "__main__":
    main()