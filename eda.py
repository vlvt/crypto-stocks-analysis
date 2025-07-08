import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class EDA:
    def __init__(self, df: pd.DataFrame, vol_window: int = 30):
            self.df = df.copy()

            try:
                if isinstance(self.df.columns, pd.MultiIndex):
                    self.df.columns = self.df.columns.droplevel(1)
                    print("MultiIndex detected and dropped.")
            except Exception as e:
                print("No MultiIndex to drop or error occurred:", e)

            self.vol_window = vol_window

    def show_head(self, n=5):
        print("-- DataFrame Head --")
        print(self.df.head(n))

    def show_shapes_and_types(self):
        print("-- DataFrame Shapes and Types --")
        print(f"Shape: {self.df.shape}")
        print(f"Data Types:\n{self.df.dtypes}")

    def show_missing_values(self):
        missing = self.df.isnull().sum()
        print("Missing values by column:\n", missing[missing > 0])

    def compute_returns(self):
        self.df['daily_ret'] = self.df['Close'].pct_change() * 100
        self.df['log_ret']   = np.log(self.df['Close'] / self.df['Close'].shift(1)) * 100

    def compute_volatility(self):
        col = f'roll_vol_{self.vol_window}'
        self.df[col] = self.df['log_ret'].rolling(self.vol_window).std()


    def compute_cumulative(self):
        self.df['cum_ret'] = (1 + self.df['daily_ret'] / 100).cumprod() - 1
        self.df['cum_ret'] = (1 + self.df['daily_ret'] / 100).cumprod() - 1


    def compute_correlation_matrix(self):
        df_cleaned = self.df.dropna()
        returns = df_cleaned.corr()

    def plot_correlation_matrix(self):

        corr = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix Heatmap')
        plt.show()

    def plot_dynamic_correlation(self, ticker: str):
            """
            Split by rolling volatility median into High/Low regimes
            and plot correlation heatmaps for each.
            """
            vol_col = f'roll_vol_{self.vol_window}'    # e.g. 'roll_vol_30'
            df = self.df.dropna()
            
            # 1) Compute median
            median_vol = df[vol_col].median()
            print(f"Median volatility is: {median_vol:.4f}")
            
            # 2) Split regimes
            high_vol = df[df[vol_col] >  median_vol]
            low_vol  = df[df[vol_col] <= median_vol]
            
            # 3) Correlation for each
            for name, subset in [("High Volatility", high_vol), ("Low Volatility", low_vol)]:
                corr = subset.corr()
                print(f"\nCorrelation ({name} periods):\n", corr)
                
                plt.figure(figsize=(10,8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title(f"{ticker} Correlation Heatmap – {name}")
                plt.show()

    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns



    def compute_volatility_regimes(self, n_clusters: int = 3): 
        vol_col = f'roll_vol_{self.vol_window}'
        vol_df = self.df[[vol_col]].dropna()

        # KMeans for volatility
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        regimes = kmeans.fit_predict(vol_df)

        # to initial df
        self.df.loc[vol_df.index, 'vol_regime'] = regimes
        return regimes

    def plot_volatility_regimes(self, ticker: str):
        
        vol_col = f'roll_vol_{self.vol_window}'
        df = self.df.dropna(subset=['vol_regime'])

        # 1) Price + regime scatter
        plt.figure(figsize=(12,6))
        plt.plot(self.df['Close'], color='grey', alpha=0.5)
        for regime in sorted(df['vol_regime'].unique()):
            idx = df[df['vol_regime']==regime].index
            plt.scatter(idx,
                        self.df.loc[idx, 'Close'],
                        s=10,
                        label=f'Regime {regime}')
        plt.title(f'{ticker} Price with Volatility Regimes')
        plt.xlabel('Date')
        plt.ylabel('Price, USD')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2) Countplot regimes
        plt.figure(figsize=(8,5))
        sns.countplot(x='vol_regime', data=df)
        plt.title('Number of Days in Each Volatility Regime')
        plt.xlabel('Volatility Regime')
        plt.ylabel('Count of Days')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def run_all(self) -> pd.DataFrame:
            self.compute_returns()
            self.compute_volatility()
            self.compute_cumulative()
            return self.df


