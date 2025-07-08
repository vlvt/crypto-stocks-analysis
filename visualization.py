import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm

# Default figure size for all plots
DEFAULT_FIGSIZE = (12, 6)

def plot_time_series(df, ticker: str, vol_window: int):

    configs = [
        ('Close', f'Historical {ticker} Price',  'Price, USD'),
        ('daily_ret','Daily Return (%)','Return, %'),
        (f'roll_vol_{vol_window}', f'Rolling {vol_window}-Day Volatility', 'Volatility, %'),
        ('cum_ret', 'Cumulative Return', 'Cum. Return'),
    ]

    for column, title, ylabel in configs:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.plot(df.index, df[column])
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        plt.show()

def plot_return_histogram(df, ticker: str):
    """
    Plot histogram of daily returns for a given ticker.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.histplot(df['daily_ret'].dropna(), bins=50, ax=ax, edgecolor='black')
    ax.set_title(f'Histogram of Daily Returns for {ticker}')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    plt.show()

def plot_return_distribution(
    daily_returns: pd.Series,
    ticker: str,
    figsize: tuple = (12,6)
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        daily_returns.dropna(),
        bins=50,
        stat='density',
        edgecolor='black',
        ax=ax,
        label='Histogram'
    )
    x = np.linspace(*ax.get_xlim(), 200)
    p = norm.pdf(x, daily_returns.mean(), daily_returns.std())
    ax.plot(x, p, 'r-', lw=2, label='Normal PDF')
    ax.set_title(f'Histogram of Daily Returns for {ticker}')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    plt.show()



def plot_monte_carlo(
    sim: dict,
    days: int,
    ticker: str,
    figsize: tuple = (12,6)
):
    paths       = sim['paths']
    median_path = sim['median_path']
    lower       = sim['lower_bound']
    upper       = sim['upper_bound']


    fig, ax = plt.subplots(figsize=figsize)
    for path in paths[:20]:
        ax.plot(path, alpha=0.3, color='grey')

    ax.plot(median_path, color='blue', label='Median Path')
    ax.plot(lower,       color='red',   linestyle='--', label='5th Percentile')
    ax.plot(upper,       color='green', linestyle='--', label='95th Percentile')

    ax.set_title(f'Monte Carlo Simulation of {ticker} Price')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.legend()
    ax.grid(True)
    plt.show()


    print(f"Probability of loss over {days} days: {sim['prob_loss']:.2f}%")
