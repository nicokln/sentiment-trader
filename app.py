import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.fetch_prices import get_stock_data
from data.fetch_news import get_news_sentiment
from data.fetch_reddit import get_reddit_sentiment
from models.sentiment_aggregator import aggregate_sentiment
from models.signal_generator import generate_signal
from models.backtest import run_backtest
from utils.metrics import compute_metrics

st.set_page_config(
    page_title="SentimentEdge",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #111318;
    --surface2: #1a1d24;
    --accent: #00f5a0;
    --accent2: #00d4ff;
    --red: #ff4d6d;
    --text: #e8eaf0;
    --muted: #6b7280;
    --border: #232730;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

.stApp { background: var(--bg) !important; }

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label { font-size: 11px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 800; font-family: 'Space Mono', monospace; }
.metric-delta { font-size: 12px; font-family: 'Space Mono', monospace; margin-top: 4px; }
.positive { color: var(--accent); }
.negative { color: var(--red); }
.neutral { color: var(--accent2); }

.signal-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 1px;
}
.signal-buy { background: rgba(0,245,160,0.15); color: var(--accent); border: 1px solid var(--accent); }
.signal-sell { background: rgba(255,77,109,0.15); color: var(--red); border: 1px solid var(--red); }
.signal-hold { background: rgba(0,212,255,0.15); color: var(--accent2); border: 1px solid var(--accent2); }

.section-header {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

div[data-testid="stSelectbox"] > div, 
div[data-testid="stMultiSelect"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}

.stSlider > div { color: var(--accent) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0c10 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}

.ticker-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--muted);
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 SentimentEdge")
    st.markdown('<p class="ticker-header">NLP · S&P 500 · Trading Signal</p>', unsafe_allow_html=True)
    st.divider()

    ticker = st.selectbox(
        "Ticker",
        ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "JPM"],
        index=0
    )

    lookback = st.slider("Lookback (jours)", 30, 180, 90, step=10)

    st.markdown("**Sources de données**")
    use_news = st.checkbox("📰 News financières", value=True)
    use_reddit = st.checkbox("🤖 Reddit WSB", value=True)

    signal_threshold = st.slider("Seuil signal sentiment", 0.0, 1.0, 0.6, step=0.05,
                                  help="Score min pour déclencher un signal BUY/SELL")

    st.divider()
    run = st.button("⚡ Analyser", use_container_width=True)

    st.markdown("""
    <div style='margin-top: 32px; font-size: 11px; color: #6b7280; line-height: 1.8;'>
    <b>Modèle</b> FinBERT (ProsusAI)<br>
    <b>Données prix</b> yfinance<br>
    <b>Reddit</b> PRAW API<br>
    <b>News</b> NewsAPI.org
    </div>
    """, unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"# {ticker} — Analyse de Sentiment")
    st.markdown(f'<span class="ticker-header">Fenêtre : {lookback} jours · Mise à jour : {datetime.now().strftime("%d %b %Y %H:%M")}</span>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# ─── MAIN LOGIC ───────────────────────────────────────────────────────────────
if run:
    with st.spinner("Récupération des données..."):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback)

        prices_df = get_stock_data(ticker, start_date, end_date)

        news_df = pd.DataFrame()
        reddit_df = pd.DataFrame()

        if use_news:
            news_df = get_news_sentiment(ticker, start_date, end_date)
        if use_reddit:
            reddit_df = get_reddit_sentiment(ticker, start_date, end_date)

        sentiment_df = aggregate_sentiment(news_df, reddit_df, prices_df)
        signal_df = generate_signal(sentiment_df, threshold=signal_threshold)
        backtest_df, perf = run_backtest(signal_df, prices_df)
        metrics = compute_metrics(backtest_df)

    # ── Signal du jour ────────────────────────────────────────────────────────
    latest = signal_df.iloc[-1]
    signal_today = latest.get('signal', 'HOLD')
    sentiment_score = latest.get('sentiment_score', 0.0)
    current_price = prices_df['Close'].iloc[-1]
    price_change = (prices_df['Close'].iloc[-1] / prices_df['Close'].iloc[-2] - 1) * 100

    badge_class = {'BUY': 'signal-buy', 'SELL': 'signal-sell', 'HOLD': 'signal-hold'}.get(signal_today, 'signal-hold')
    score_color = 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Signal Aujourd'hui</div>
            <div style="margin-top: 8px;"><span class="signal-badge {badge_class}">{signal_today}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Score Sentiment</div>
            <div class="metric-value {score_color}">{sentiment_score:+.3f}</div>
            <div class="metric-delta {score_color}">{'Bullish' if sentiment_score > 0.1 else 'Bearish' if sentiment_score < -0.1 else 'Neutre'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        pc = 'positive' if price_change > 0 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prix Actuel</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-delta {pc}">{price_change:+.2f}% aujourd'hui</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sharpe = metrics.get('sharpe', 0)
        sc = 'positive' if sharpe > 1 else 'negative' if sharpe < 0 else 'neutral'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio (stratégie)</div>
            <div class="metric-value {sc}">{sharpe:.2f}</div>
            <div class="metric-delta">vs B&H: {metrics.get('sharpe_bh', 0):.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Prix & Signaux", "🧠 Sentiment", "⚙️ Backtest"])

    PLOTLY_LAYOUT = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf0', family='Space Mono'),
        xaxis=dict(gridcolor='#1a1d24', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#1a1d24', showgrid=True, zeroline=False),
        margin=dict(t=30, b=30, l=10, r=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#232730')
    )

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.04)

        fig.add_trace(go.Candlestick(
            x=prices_df.index,
            open=prices_df['Open'], high=prices_df['High'],
            low=prices_df['Low'], close=prices_df['Close'],
            name=ticker,
            increasing_line_color='#00f5a0', decreasing_line_color='#ff4d6d',
            increasing_fillcolor='rgba(0,245,160,0.3)', decreasing_fillcolor='rgba(255,77,109,0.3)'
        ), row=1, col=1)

        buys = signal_df[signal_df['signal'] == 'BUY']
        sells = signal_df[signal_df['signal'] == 'SELL']

        if not buys.empty:
            buy_prices = prices_df.loc[prices_df.index.isin(buys.index), 'Low'] * 0.99
            fig.add_trace(go.Scatter(
                x=buy_prices.index, y=buy_prices.values,
                mode='markers', name='Signal BUY',
                marker=dict(symbol='triangle-up', size=12, color='#00f5a0', line=dict(color='#0a0c10', width=1))
            ), row=1, col=1)

        if not sells.empty:
            sell_prices = prices_df.loc[prices_df.index.isin(sells.index), 'High'] * 1.01
            fig.add_trace(go.Scatter(
                x=sell_prices.index, y=sell_prices.values,
                mode='markers', name='Signal SELL',
                marker=dict(symbol='triangle-down', size=12, color='#ff4d6d', line=dict(color='#0a0c10', width=1))
            ), row=1, col=1)

        vol_colors = ['#00f5a0' if c >= o else '#ff4d6d'
                      for c, o in zip(prices_df['Close'], prices_df['Open'])]
        fig.add_trace(go.Bar(
            x=prices_df.index, y=prices_df['Volume'],
            name='Volume', marker_color=vol_colors, opacity=0.5
        ), row=2, col=1)

        fig.update_layout(**PLOTLY_LAYOUT, height=520)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not sentiment_df.empty:
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.6, 0.4], vertical_spacing=0.06,
                                 subplot_titles=["Score Sentiment Agrégé", "Volume de mentions"])

            colors_sent = ['#00f5a0' if s > 0 else '#ff4d6d' for s in sentiment_df['sentiment_score']]
            fig2.add_trace(go.Bar(
                x=sentiment_df.index, y=sentiment_df['sentiment_score'],
                name='Sentiment', marker_color=colors_sent, opacity=0.8
            ), row=1, col=1)

            if 'sentiment_ma' in sentiment_df.columns:
                fig2.add_trace(go.Scatter(
                    x=sentiment_df.index, y=sentiment_df['sentiment_ma'],
                    name='MA 7j', line=dict(color='#00d4ff', width=2)
                ), row=1, col=1)

            if 'mention_count' in sentiment_df.columns:
                fig2.add_trace(go.Bar(
                    x=sentiment_df.index, y=sentiment_df['mention_count'],
                    name='Mentions', marker_color='#00d4ff', opacity=0.6
                ), row=2, col=1)

            fig2.update_layout(**PLOTLY_LAYOUT, height=450)
            st.plotly_chart(fig2, use_container_width=True)

            if 'source' in sentiment_df.columns:
                source_counts = sentiment_df['source'].value_counts()
                fig_pie = px.pie(values=source_counts.values, names=source_counts.index,
                                 color_discrete_sequence=['#00f5a0', '#00d4ff', '#ff4d6d'])
                fig_pie.update_layout(**PLOTLY_LAYOUT, height=300, title="Répartition des sources")
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Aucune donnée de sentiment disponible.")

    with tab3:
        fig3 = go.Figure()
        if 'cumret_strategy' in backtest_df.columns:
            fig3.add_trace(go.Scatter(
                x=backtest_df.index, y=backtest_df['cumret_strategy'] * 100,
                name='Stratégie Sentiment', line=dict(color='#00f5a0', width=2.5),
                fill='tozeroy', fillcolor='rgba(0,245,160,0.06)'
            ))
        if 'cumret_bh' in backtest_df.columns:
            fig3.add_trace(go.Scatter(
                x=backtest_df.index, y=backtest_df['cumret_bh'] * 100,
                name='Buy & Hold', line=dict(color='#00d4ff', width=2, dash='dot')
            ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=380,
                           yaxis_title="Performance cumulée (%)",
                           title="Stratégie Sentiment vs Buy & Hold")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<p class="section-header">Métriques de Performance</p>', unsafe_allow_html=True)
        mcols = st.columns(5)
        metric_items = [
            ("Perf. Totale", f"{metrics.get('total_return', 0)*100:.1f}%", metrics.get('total_return', 0) > 0),
            ("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}", metrics.get('sharpe', 0) > 1),
            ("Max Drawdown", f"{metrics.get('max_dd', 0)*100:.1f}%", False),
            ("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%", metrics.get('win_rate', 0) > 0.5),
            ("Nb Trades", f"{metrics.get('nb_trades', 0)}", True),
        ]
        for col, (label, value, positive) in zip(mcols, metric_items):
            color = 'positive' if positive else 'negative' if label != 'Nb Trades' else 'neutral'
            col.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}" style="font-size:20px;">{value}</div>
            </div>""", unsafe_allow_html=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px;">
        <div style="font-size: 64px; margin-bottom: 16px;">📡</div>
        <h2 style="font-size: 32px; margin-bottom: 12px;">NLP + Trading Signal</h2>
        <p style="color: #6b7280; font-size: 16px; max-width: 500px; margin: 0 auto; line-height: 1.7;">
            Sélectionne un ticker, configure tes sources de données<br>
            et clique sur <b style="color: #00f5a0;">Analyser</b> pour générer les signaux.
        </p>
        <div style="display: flex; justify-content: center; gap: 32px; margin-top: 48px; flex-wrap: wrap;">
            <div style="background: #111318; border: 1px solid #232730; border-radius: 12px; padding: 24px 32px;">
                <div style="font-size: 28px;">🧠</div>
                <div style="font-size: 13px; color: #6b7280; margin-top: 8px;">FinBERT NLP</div>
            </div>
            <div style="background: #111318; border: 1px solid #232730; border-radius: 12px; padding: 24px 32px;">
                <div style="font-size: 28px;">📰</div>
                <div style="font-size: 13px; color: #6b7280; margin-top: 8px;">News + Reddit</div>
            </div>
            <div style="background: #111318; border: 1px solid #232730; border-radius: 12px; padding: 24px 32px;">
                <div style="font-size: 28px;">⚙️</div>
                <div style="font-size: 13px; color: #6b7280; margin-top: 8px;">Backtest intégré</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
