# src/main.py

import numpy as np

from .config import TICKERS, START_DATE, END_DATE, INTERVAL, VOL_WINDOW, CONFIDENCE_LEVEL
from .data_ingest import load_data
from .eda import EDA
from .visualization import (
    plot_time_series,
    plot_return_histogram,
    plot_return_distribution,
    plot_monte_carlo
)
from .analysis import dynamic_corr_heatmaps
from .metrics import (
    calculate_sharpe_ratio,
    calculate_cumulative_max,
    calculate_max_drawdown,
    calculate_historical_var,
    calculate_cvar
)
from .models import (
    walk_forward_forecast,
    evaluate_forecasts,
    test_stationarity,
    fit_arima,
    forecast_arima,
    simulate_monte_carlo
)

# Modeling Params
SPECS = [
    {'vol':'Garch','p':1,'q':1,'dist':'normal'},
    {'vol':'Garch','p':1,'q':1,'dist':'t'},
    {'vol':'EGarch','p':1,'q':1,'dist':'t'},
    {'vol':'Garch','p':2,'q':1,'dist':'t'},
]
FORECAST_STEPS = 5
MC_DAYS        = 30
MC_SIMS        = 1000


def main():
    for ticker in TICKERS:
        print(f"\n\n===== Processing {ticker} =====")

        # 1) Loading & EDA
        df = load_data(ticker, START_DATE, END_DATE, INTERVAL)
        eda = EDA(df, vol_window=VOL_WINDOW)
        eda.show_head()
        eda.show_shapes_and_types()
        eda.show_missing_values()
        df = eda.run_all()

        # 2) Basic Visualization
        plot_time_series(df, ticker, VOL_WINDOW)
        plot_return_histogram(df, ticker)
        plot_return_distribution(df['daily_ret'], ticker)

        # 3) Correlation and regimes
        eda.plot_correlation_matrix()
        eda.plot_dynamic_correlation(ticker)
        eda.compute_volatility_regimes(n_clusters=3)
        eda.plot_volatility_regimes(ticker)
        dynamic_corr_heatmaps(df, VOL_WINDOW, ticker)

        # 4) Risk-metrics
        var   = calculate_historical_var(df['daily_ret'], CONFIDENCE_LEVEL)
        cvar  = calculate_cvar(df['daily_ret'], CONFIDENCE_LEVEL)
        sharpe= calculate_sharpe_ratio(df['daily_ret'])
        cum_max = calculate_cumulative_max(df['cum_ret'])
        max_dd = calculate_max_drawdown(df['cum_ret'])

        print("\n--- Risk Metrics ---")
        print(f"VaR   (@{int(CONFIDENCE_LEVEL*100)}%): {var:.2f}%")
        print(f"CVaR  (@{int(CONFIDENCE_LEVEL*100)}%): {cvar:.2f}%")
        print(f"Sharpe Ratio:        {sharpe:.2f}")
        print(f"Cumulative Max:      {cum_max.iloc[-1]:.2f}")
        print(f"Max Drawdown:        {max_dd:.2%}")

        # 5) Modeling
        
        # 5a) GARCH-backtest
        returns_series = df['daily_ret'].dropna()
        split_idx = int(len(returns_series) * 0.8)
        vol_series = df[f'roll_vol_{VOL_WINDOW}']
        aligned_vol = vol_series.reindex(returns_series.index)
        actual_vol = aligned_vol.iloc[split_idx:].to_numpy(dtype=float)

        for spec in SPECS:
            label = f"{spec['vol']}({spec['p']},{spec['q']})[{spec['dist']}]"
        
        forecasts = walk_forward_forecast(returns_series, split_idx, spec)
        forecasts = np.asarray(forecasts, dtype=float)

        assert len(forecasts) == len(actual_vol), (
            f"Length mismatch: forecasts={len(forecasts)}, actual_vol={len(actual_vol)}"
        )

        mse = evaluate_forecasts(actual_vol, forecasts)
        print(f"{label} MSE: {mse:.4f}")

        # 5b) ARIMA
        print("\n--- ARIMA Forecast ---")
        stat, p, is_stat = test_stationarity(df['Close'])
        print(f"ADF test p-value={p:.3f}, stationary={is_stat}")
        arima_fit = fit_arima(df['Close'], order=(1,1,1))
        print(arima_fit.summary())
        arima_fc  = forecast_arima(arima_fit, steps=FORECAST_STEPS)
        print("Next steps forecast:\n", arima_fc.to_list())

        # 5c) Monte Carlo
        print("\n--- Monte Carlo Simulation ---")
        df_returns = df['Close'].pct_change().dropna()
        mu, sigma  = df_returns.mean(), df_returns.std()
        last_price = df['Close'].iloc[-1]

        sim = simulate_monte_carlo(
            last_price=last_price,
            mu=mu,
            sigma=sigma,
            days=MC_DAYS,
            simulations=MC_SIMS,
            seed=42
        )
        plot_monte_carlo(sim, days=MC_DAYS, ticker=ticker)


if __name__ == "__main__":
    main()
