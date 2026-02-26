import pandas as pd
import numpy as np


def generate_signal(
    sentiment_df: pd.DataFrame,
    threshold: float = 0.6,
    ma_window: int = 7,
    use_momentum: bool = True
) -> pd.DataFrame:
    """
    Génère des signaux de trading à partir du score de sentiment.

    Logique :
    ---------
    - On utilise le score MA pour filtrer le bruit
    - BUY  : sentiment_ma > +threshold  ET momentum positif (si activé)
    - SELL : sentiment_ma < -threshold  ET momentum négatif
    - HOLD : sinon

    Le threshold est normalisé par rapport à l'étendue historique du score.

    Paramètres
    ----------
    sentiment_df   : sortie de aggregate_sentiment()
    threshold      : seuil absolu sur le score (entre 0 et 1)
    ma_window      : fenêtre de lissage
    use_momentum   : filtre sur le changement de sentiment (évite les faux signaux)

    Retourne
    --------
    DataFrame avec colonnes : sentiment_score, sentiment_ma, signal
    """
    df = sentiment_df.copy()

    # Recalcul MA au cas où
    df['sentiment_ma'] = df['sentiment_score'].rolling(ma_window, min_periods=1).mean()

    # Momentum : variation du score sur 3 jours
    df['sentiment_momentum'] = df['sentiment_score'].diff(3)

    # Normalisation rolling (z-score fenêtre 30 jours)
    roll_mean = df['sentiment_score'].rolling(30, min_periods=5).mean()
    roll_std = df['sentiment_score'].rolling(30, min_periods=5).std().replace(0, 1)
    df['sentiment_zscore'] = (df['sentiment_score'] - roll_mean) / roll_std

    # Condition de signal
    bull_condition = df['sentiment_ma'] > threshold
    bear_condition = df['sentiment_ma'] < -threshold

    if use_momentum:
        bull_condition &= df['sentiment_momentum'] > 0
        bear_condition &= df['sentiment_momentum'] < 0

    df['signal'] = 'HOLD'
    df.loc[bull_condition, 'signal'] = 'BUY'
    df.loc[bear_condition, 'signal'] = 'SELL'

    # Filtre : ne pas répéter le même signal consécutif (évite le surtrading)
    prev_signal = df['signal'].shift(1)
    df.loc[(df['signal'] == prev_signal) & (df['signal'] != 'HOLD'), 'signal'] = 'HOLD'

    return df[['sentiment_score', 'sentiment_ma', 'sentiment_zscore', 'signal']]
