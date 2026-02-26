import pandas as pd
import numpy as np


def aggregate_sentiment(
    news_df: pd.DataFrame,
    reddit_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    news_weight: float = 0.6,
    reddit_weight: float = 0.4
) -> pd.DataFrame:
    """
    Fusionne le sentiment news et Reddit en un score journalier unique.

    Paramètres
    ----------
    news_df      : DataFrame avec index date et colonne 'sentiment_score'
    reddit_df    : idem
    prices_df    : DataFrame prix (sert de calendrier de référence)
    news_weight  : poids des news dans l'agrégation
    reddit_weight: poids de Reddit

    Retourne
    --------
    DataFrame avec colonnes :
      sentiment_score, sentiment_ma, mention_count, source
    """
    # Calendrier de référence = jours de bourse
    cal = pd.DataFrame(index=prices_df.index)

    frames = []

    if not news_df.empty and 'sentiment_score' in news_df.columns:
        n = news_df[['sentiment_score']].rename(columns={'sentiment_score': 'news_score'})
        n['news_count'] = news_df.get('mention_count', pd.Series(1, index=news_df.index))
        frames.append(('news', n, news_weight))

    if not reddit_df.empty and 'sentiment_score' in reddit_df.columns:
        r = reddit_df[['sentiment_score']].rename(columns={'sentiment_score': 'reddit_score'})
        r['reddit_count'] = reddit_df.get('mention_count', pd.Series(1, index=reddit_df.index))
        frames.append(('reddit', r, reddit_weight))

    merged = cal.copy()

    total_weight = 0
    weighted_scores = pd.Series(0.0, index=cal.index)

    for name, frame, weight in frames:
        # Reindex sur le calendrier bourse, forward-fill max 3 jours
        reindexed = frame.reindex(cal.index).ffill(limit=3).fillna(0)
        col = f"{name}_score"
        merged[col] = reindexed.iloc[:, 0]
        weighted_scores += reindexed.iloc[:, 0] * weight
        total_weight += weight

        count_col = f"{name}_count"
        merged[count_col] = reindexed.iloc[:, 1] if reindexed.shape[1] > 1 else 0

    if total_weight > 0:
        merged['sentiment_score'] = weighted_scores / total_weight
    else:
        merged['sentiment_score'] = 0.0

    # Moving average 7 jours
    merged['sentiment_ma'] = merged['sentiment_score'].rolling(7, min_periods=1).mean()

    # Mention count total
    count_cols = [c for c in merged.columns if c.endswith('_count')]
    merged['mention_count'] = merged[count_cols].sum(axis=1) if count_cols else 0

    # Source dominante
    if 'news_score' in merged.columns and 'reddit_score' in merged.columns:
        merged['source'] = 'news+reddit'
    elif 'news_score' in merged.columns:
        merged['source'] = 'news'
    else:
        merged['source'] = 'reddit'

    return merged[['sentiment_score', 'sentiment_ma', 'mention_count', 'source']]
