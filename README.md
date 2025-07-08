# 📈 Asset Volatility & Price-Forecasting Framework

This repository contains a **modular**, **plug-and-play** pipeline for loading, exploring, visualizing, and forecasting the volatility, returns and risk metrics of **any financial asset**—from equities and commodities to cryptocurrencies. Simply specify your ticker (e.g. `BTC-USD`, `ETH-USD`, `AAPL`, `SPY`), date range and interval, and the framework will:

- Ingest historical price data (via Yahoo Finance or any compatible source)  
- Compute **log-returns**, **cumulative returns**, rolling **volatility**, and **regime clusters**  
- Calculate **risk metrics**: VaR, CVaR, Sharpe ratio, maximum drawdown  
- Fit and backtest **ARIMA**, **GARCH/EGARCH**, and **Monte Carlo** models  
- Produce publication-quality visualizations with minimal code  

> **Why this project?**  
> - **Flexibility**: swap in new models (e.g. EWMA volatility), metrics or data sources (EHO Finances, Alpha Vantage)  
> - **Reproducibility**: one command runs the full EDA, risk analysis & forecasting pipeline  
> - **Extensibility**: add hyperparameter tuning, live data feeds, CI/CD or interactive dashboards (Plotly Dash, Streamlit)

---

## 🧠 Project Objectives

1. **Data ingestion & EDA**  
   - Load OHLC data for any ticker and interval  
   - Compute and visualize log-returns:  
   - Rolling volatility over window:  
   - Cluster volatility regimes via K-means 

2. **Risk metrics**  
   - **VaR** at confidence level    
   - **CVaR** (Conditional VaR)  
   - **Sharpe ratio**  
   - **Max drawdown** 

3. **Forecasting & backtesting**  
   - **GARCH/EGARCH** (walk-forward backtest): forecast next-day volatility, evaluate MSE  
   - **ARIMA**:  
     - ADF test for stationarity (unit-root),  
     - Fit ARIMA\((p,d,q)\),  
     - Forecast \(h\) steps ahead  
   - **Monte Carlo** (Geometric Brownian Motion)

4. **Visualization**  
   - Time series of prices & volatility  
   - Histograms & KDE of returns  
   - Dynamic correlation heatmaps  
   - Monte Carlo scenario  

---

---

## ⚙️ Installation

1. **Clone repository**  
   ```bash
   git clone https://github.com/your_username/asset-forecasting-framework.git
   cd asset-forecasting-framework
2. **Install**
   pip install -r requirements.txt
3. **optional**
   python -m venv venv
  source venv/bin/activate   # on Linux/macOS
  venv\Scripts\activate      # on Windows
  pip install -r requirements.txt


## 🚀 Usage

1. **Configure parameters** in `src/config.py`:  
   ```python
   TICKERS = ['BTC-USD']       # List of assets
   START_DATE = '2021-01-01'   # Start date
   END_DATE = '2025-06-09'     # End date
   INTERVAL = '1d'             # Interval ('1d', '1h', ...)
   VOL_WINDOW = 30             # Rolling window for volatility calculation
   CONFIDENCE_LEVEL = 0.95     # Confidence level for VaR/CVaR

 2. **Run the full analysis**
   ```python 
    python -m src.main
