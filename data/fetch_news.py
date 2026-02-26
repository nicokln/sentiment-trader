import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import requests

# ── Chargement lazy du modèle FinBERT ─────────────────────────────────────────
_finbert = None

def _get_finbert():
    global _finbert
    if _finbert is None:
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1  # CPU; remplace par 0 si tu as un GPU
        )
    return _finbert


def _score_text(text: str) -> float:
    """
    Retourne un score entre -1 (très bearish) et +1 (très bullish).
    FinBERT classe en : positive / negative / neutral.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        finbert = _get_finbert()
        result = finbert(text[:512])[0]  # limite tokens
        scores = {r['label']: r['score'] for r in result}
        score = scores.get('positive', 0) - scores.get('negative', 0)
        return round(float(score), 4)
    except Exception:
        return 0.0


def get_news_sentiment(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Récupère les articles via NewsAPI et calcule leur score de sentiment.

    Nécessite : NEWSAPI_KEY dans les variables d'environnement.
    Si pas de clé → génère des données simulées pour démonstration.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")

    articles = []

    if api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": api_key
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            for article in data.get("articles", []):
                text = f"{article.get('title', '')} {article.get('description', '')}"
                published = article.get("publishedAt", "")[:10]
                score = _score_text(text)
                articles.append({
                    "date": pd.to_datetime(published),
                    "text": text[:200],
                    "sentiment_score": score,
                    "source": "news"
                })
        except Exception as e:
            print(f"NewsAPI error: {e}")

    # ── Fallback : données simulées réalistes ─────────────────────────────────
    if not articles:
        articles = _simulate_news_sentiment(ticker, start_date, end_date)

    df = pd.DataFrame(articles)
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date', 'sentiment_score'])
    df = df.set_index('date').sort_index()

    # Agrégation journalière : moyenne pondérée
    daily = df.groupby(df.index.date).agg(
        sentiment_score=('sentiment_score', 'mean'),
        mention_count=('sentiment_score', 'count'),
        source=('source', 'first')
    )
    daily.index = pd.to_datetime(daily.index)

    return daily.reset_index().rename(columns={'index': 'date'}).set_index('date')


def _simulate_news_sentiment(ticker: str, start_date: datetime, end_date: datetime) -> list:
    """
    Génère des données de sentiment simulées avec autocorrélation
    (pour demo sans clé API).
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(hash(ticker) % 1000)

    # AR(1) process pour le sentiment
    scores = []
    s = 0.0
    for _ in dates:
        s = 0.6 * s + 0.4 * np.random.normal(0, 0.4)
        s = np.clip(s, -1, 1)
        scores.append(round(s, 4))

    return [
        {
            "date": d,
            "text": f"[Simulated] News about {ticker}",
            "sentiment_score": s,
            "source": "news (simulated)"
        }
        for d, s in zip(dates, scores)
    ]
