import os
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import pipeline

# Réutilise le même modèle FinBERT (lazy load partagé)
_finbert = None

def _get_finbert():
    global _finbert
    if _finbert is None:
        from transformers import pipeline as pl
        _finbert = pl(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1
        )
    return _finbert


def _score_text(text: str) -> float:
    if not text or len(text.strip()) < 10:
        return 0.0
    try:
        result = _get_finbert()(text[:512])[0]
        scores = {r['label']: r['score'] for r in result}
        return round(float(scores.get('positive', 0) - scores.get('negative', 0)), 4)
    except Exception:
        return 0.0


def get_reddit_sentiment(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Récupère les posts Reddit (r/wallstreetbets, r/stocks, r/investing)
    et calcule leur score de sentiment.

    Nécessite dans les variables d'environnement :
      REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

    Sans credentials → données simulées pour démonstration.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "SentimentTrader/1.0")

    posts = []

    if client_id and client_secret:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            subreddits = ["wallstreetbets", "stocks", "investing"]
            query = f"{ticker} stock"

            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.search(query, sort="new", time_filter="month", limit=50):
                    post_date = datetime.fromtimestamp(post.created_utc)
                    if start_date <= post_date <= end_date:
                        text = f"{post.title} {post.selftext[:300]}"
                        score = _score_text(text)
                        posts.append({
                            "date": post_date.date(),
                            "text": post.title[:200],
                            "sentiment_score": score,
                            "upvotes": post.score,
                            "source": f"reddit/{sub_name}"
                        })
        except Exception as e:
            print(f"Reddit API error: {e}")

    # ── Fallback simulé ───────────────────────────────────────────────────────
    if not posts:
        posts = _simulate_reddit_sentiment(ticker, start_date, end_date)

    df = pd.DataFrame(posts)
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date', 'sentiment_score'])
    df = df.set_index('date').sort_index()

    # Agrégation journalière (pondérée par upvotes si disponible)
    def weighted_avg(group):
        weights = group.get('upvotes', pd.Series([1] * len(group))).clip(lower=1)
        return np.average(group['sentiment_score'], weights=weights)

    daily = df.groupby(df.index.date).apply(
        lambda g: pd.Series({
            'sentiment_score': np.average(g['sentiment_score'],
                                          weights=g.get('upvotes', pd.Series([1]*len(g))).clip(lower=1).values),
            'mention_count': len(g),
            'source': 'reddit'
        })
    )
    daily.index = pd.to_datetime(daily.index)

    return daily


def _simulate_reddit_sentiment(ticker: str, start_date: datetime, end_date: datetime) -> list:
    """Génère des données Reddit simulées avec plus de bruit que les news."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed((hash(ticker) + 42) % 1000)

    posts = []
    for d in dates:
        n_posts = np.random.randint(2, 15)
        s_base = 0.3 * np.random.normal(0, 0.5)
        for _ in range(n_posts):
            s = s_base + np.random.normal(0, 0.3)
            posts.append({
                "date": d,
                "text": f"[Simulated] Reddit post about ${ticker}",
                "sentiment_score": round(np.clip(s, -1, 1), 4),
                "upvotes": int(np.random.exponential(50)),
                "source": "reddit (simulated)"
            })

    return posts
