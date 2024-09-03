# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_comparison(df, metric, top_n=10):
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values(metric, ascending=True)
    ax = sns.barplot(x=df_sorted[metric][:top_n], y=df_sorted.index[:top_n])
    plt.title(f'Top {top_n} Models by {metric}')
    plt.xlabel(metric)
    plt.tight_layout()
    
    for i, v in enumerate(df_sorted[metric][:top_n]):
        ax.text(v, i, f' {v:.4f}', va='center')
    
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Metrics')
    plt.tight_layout()
    plt.show()

def plot_r2_vs_time(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Computation Time (s)', y='R2', data=df)
    for i, model in enumerate(df.index):
        plt.annotate(model, (df['Computation Time (s)'][i], df['R2'][i]))
    plt.title('R2 Score vs Computation Time')
    plt.tight_layout()
    plt.show()