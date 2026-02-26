import yfinance as yf
import pandas as pd
from datetime import datetime


def get_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis Yahoo Finance.
    Retourne un DataFrame avec colonnes : Open, High, Low, Close, Volume.
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"Aucune donnée trouvée pour {ticker}")

    # Flatten multi-level columns si nécessaire
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)

    return df
