import yfinance as yf
import pandas as pd

def load_data(ticker: str, start: str, end: str, interval: str='1d') -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.dropna(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df
