import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any



#GARCH model fitting and forecasting functions

def fit_garch(returns: pd.Series, vol: str, p: int, q: int, dist: str):
    """Fit a single ARCH/GARCH‐family model."""
    model = arch_model(returns, vol=vol, p=p, q=q, dist=dist)
    fit   = model.fit(disp='off')
    return fit

def walk_forward_forecast(
    returns: pd.Series,
    split: int,
    spec: dict,
    horizon: int = 1
) -> np.ndarray:
    """
    Generic walk‐forward forecasting for a single model spec.
    spec = {'vol':'Garch','p':1,'q':1,'dist':'t'}
    Returns array of predicted vol for each test point.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns).dropna()
    else:
        returns = returns.dropna()
    
    forecasts = []
    for i in range(len(returns) - split):
        window = returns[: split + i]
        fit = fit_garch(window, **spec)
        fc  = fit.forecast(horizon=horizon)
        pred_vol = np.sqrt(fc.variance.values[-1, :][0])
        forecasts.append(pred_vol)
        
    return np.array(forecasts)

def evaluate_forecasts(
    actual_vol: pd.Series,
    forecasts: np.ndarray
) -> float:
    """Compute MSE between actual rolling vol and forecasted vol."""
    return mean_squared_error(actual_vol, forecasts)



 #ARIMA model fitting and forecasting functions

def test_stationarity(series, alpha: float = 0.05):
    stat, p, *_ = adfuller(series.dropna())
    return stat, p, (p <= alpha)

def fit_arima(series, order=(1,1,1)):
    model = ARIMA(series.dropna(), order=order)
    return model.fit()

def forecast_arima(fit_result, steps: int = 5):
    return fit_result.forecast(steps=steps)


#Monte Carlo simulation 

def simulate_monte_carlo(
    last_price: float,
    mu: float,
    sigma: float,
    days: int,
    simulations: int,
    seed: int = 42
) -> dict:
    """
    Monte Carlo price paths under Geometric Brownian Motion.
    Returns a dict:
      {
        'paths':         np.ndarray shape (simulations, days+1),
        'median_path':   np.ndarray length days+1,
        'lower_bound':   np.ndarray length days+1 (5th percentile),
        'upper_bound':   np.ndarray length days+1 (95th percentile),
        'prob_loss':     float (% of sims ending below start price)
      }
    """
    np.random.seed(seed)
    paths = np.zeros((simulations, days + 1))
    paths[:, 0] = last_price

    for i in range(simulations):
        rand_rets = np.random.normal(mu, sigma, days)
        for t, r in enumerate(rand_rets, start=1):
            paths[i, t] = paths[i, t-1] * (1 + r)


    median_path = np.median(paths, axis=0)
    lower_bound = np.percentile(paths, 5, axis=0)
    upper_bound = np.percentile(paths, 95, axis=0)
    final_prices = paths[:, -1]
    prob_loss = np.mean(final_prices < last_price) * 100

    return {
        'paths':       paths,
        'median_path': median_path,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'prob_loss':   prob_loss
    }
