import matplotlib.pyplot as plt
import seaborn as sns

def dynamic_corr_heatmaps(df, vol_window, ticker):
    """
    Plot correlation heatmaps for high- and low-volatility regimes.
    """
    df = df.dropna()
    vol_col = f'roll_vol_{vol_window}'
    median_vol = df[vol_col].median()
    print(f"Median volatility is: {median_vol:.4f}")
    
    high = df[df[vol_col] >  median_vol]
    low  = df[df[vol_col] <= median_vol]
    
    for name, subset in [("High Volatility", high), ("Low Volatility", low)]:
        corr = subset.corr()
        print(f"\nCorrelation ({name} periods):\n", corr)
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"{ticker} Correlation – {name}")
        plt.show()
