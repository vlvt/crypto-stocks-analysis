import numpy as np
import pandas as pd

def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    r = returns.dropna() / 100.0
    mean_ret = r.mean()
    std_ret  = r.std()

    # Annualized Sharpe
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    return sharpe


def calculate_cumulative_max(
    cum_returns: pd.Series
) -> pd.Series:

    return cum_returns.cummax()


def calculate_max_drawdown(
    cum_returns: pd.Series
) -> float:
    cum_max = cum_returns.cummax()
    drawdowns = (cum_returns - cum_max) / cum_max
    return drawdowns.min()

def calculate_historical_var(returns: pd.Series, confidence: float) -> float:
    return np.percentile(returns.dropna(), (1 - confidence) * 100)

def calculate_cvar(returns: pd.Series, confidence: float) -> float:
    var = calculate_historical_var(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if not tail.empty else var
