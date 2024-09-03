# utils.py

def print_top_models(df, metrics, top_n=3):
    for metric in metrics:
        ascending = True if metric not in ['R2', 'Adjusted R2'] else False
        top = df.sort_values(metric, ascending=ascending).head(top_n)
        print(f"\nTop {top_n} models by {metric}:")
        print(top[[metric]])

def calculate_overall_rank(df):
    for col in df.columns:
        if col in ['R2', 'Adjusted R2']:
            df[f'{col}_rank'] = df[col].rank(ascending=False)
        else:
            df[f'{col}_rank'] = df[col].rank()
    
    df['Overall_rank'] = df[[col for col in df.columns if col.endswith('_rank')]].mean(axis=1)
    return df.sort_values('Overall_rank')