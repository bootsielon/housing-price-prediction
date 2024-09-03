import joblib
from data_loader import load_data, split_data, preprocess_data
from data_analysis import print_data_info, plot_histograms, plot_categorical, plot_scatter, plot_correlation_matrix
from feature_engineering import create_features
from model_training import train_and_evaluate_models
from visualization import plot_metric_comparison, plot_correlation_heatmap, plot_r2_vs_time
from utils import print_top_models, calculate_overall_rank
import os
import pickle
import sys
from sklearn import __version__ as sklearn_version

def main():
    print(f"Training with Python version: {sys.version}")
    print(f"Training with scikit-learn version: {sklearn_version}")
        
    # Load and split data
    data = load_data()
    train_set, test_set = split_data(data)

    # Perform data analysis
    print_data_info(data)
    plot_histograms(data)
    plot_categorical(data, 'ocean_proximity')
    plot_scatter(data, 'longitude', 'latitude')
    plot_correlation_matrix(data)

    # Print the location of saved plots
    current_dir = os.getcwd()
    plots_dir = os.path.join(current_dir, 'plots')
    print(f"\nData analysis plots have been saved in the following directory:")
    print(plots_dir)
    print("The following plots were saved:")
    print("- histograms.png")
    print("- categorical_ocean_proximity.png")
    print("- scatter_longitude_vs_latitude.png")
    print("- correlation_matrix.png")

    # Feature engineering
    train_set = create_features(train_set)
    test_set = create_features(test_set)
    
    # Preprocess data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(train_set, test_set)
    
    # Save the scaler
    joblib.dump(scaler, 'robust_scaler.joblib')
    print("Scaler saved successfully.")
    
    # Train and evaluate models
    results_df, best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Save the best model using both joblib and pickle
    joblib.dump(best_model, 'best_model.joblib')
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model ({type(best_model).__name__}) saved successfully.")
    
    # Save the scaler using both joblib and pickle
    joblib.dump(scaler, 'robust_scaler.joblib')
    with open('robust_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # Save version information
    with open('version_info.txt', 'w') as f:
        f.write(f"Python version: {sys.version}\n")
        f.write(f"scikit-learn version: {sklearn_version}\n")

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
    print("\nTop models overall:")
    print(ranked_df[['RMSE', 'R2','Adjusted R2', 'MAPE', 'MAE', 'Overall_rank']].head())
    
    # Save results
    results_df.to_csv('model_comparison_results.csv')

if __name__ == "__main__":
    main()