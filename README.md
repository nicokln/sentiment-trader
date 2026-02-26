# 📡 SentimentEdge — NLP Trading Signal Dashboard

Projet de gestion de portefeuille combinant **NLP (FinBERT)** et données de marché
pour générer des signaux de trading sur les actions du S&P 500.

---

## 🏗️ Architecture

```
sentiment_trader/
├── app.py                          # Dashboard Streamlit principal
├── requirements.txt
├── .env.example                    # Template des clés API
│
├── data/
│   ├── fetch_prices.py             # Données OHLCV via yfinance
│   ├── fetch_news.py               # Articles financiers via NewsAPI + FinBERT
│   └── fetch_reddit.py             # Posts Reddit (WSB, stocks, investing) + FinBERT
│
├── models/
│   ├── sentiment_aggregator.py     # Fusion news + Reddit → score quotidien
│   ├── signal_generator.py         # BUY / SELL / HOLD à partir du score
│   └── backtest.py                 # Backtest long-only avec coûts de transaction
│
└── utils/
    └── metrics.py                  # Sharpe, Max DD, Calmar, Win Rate...
```

---

## ⚡ Installation

### 1. Cloner / créer le projet

```bash
git clone <ton-repo>
cd sentiment_trader
```

### 2. Environnement Python

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# ou
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> 💡 Le premier lancement téléchargera le modèle **FinBERT** (~440 MB).  
> Il est ensuite mis en cache automatiquement par HuggingFace.

### 3. Variables d'environnement

```bash
cp .env.example .env
# Édite .env avec tes clés API
```

Ensuite, charge le fichier `.env` avant de lancer l'app :

```bash
# Linux/Mac
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object { $k,$v = $_ -split '=',2; [Environment]::SetEnvironmentVariable($k,$v) }
```

#### 🔑 Obtenir les clés API (optionnel)

| Service | URL | Plan gratuit |
|---------|-----|--------------|
| NewsAPI | https://newsapi.org | 100 req/jour, 1 mois d'historique |
| Reddit | https://www.reddit.com/prefs/apps | Gratuit (app "script") |

> **Sans clés API** : l'application fonctionne avec des données simulées réalistes.  
> Parfait pour tester et démontrer la logique lors d'un entretien.

### 4. Lancer l'application

```bash
streamlit run app.py
```

Ouvre [http://localhost:8501](http://localhost:8501) dans ton navigateur.

---

## 🧠 Logique du Modèle

### 1. Collecte des données textuelles
- **NewsAPI** : articles financiers contenant le ticker (ex: "AAPL")
- **Reddit** : posts de r/wallstreetbets, r/stocks, r/investing

### 2. Analyse de sentiment (FinBERT)
[ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) est un modèle BERT
fine-tuné sur des données financières. Il classifie chaque texte en :
- `positive` → bullish
- `negative` → bearish  
- `neutral`

**Score** = P(positive) - P(negative) ∈ [-1, +1]

### 3. Agrégation du signal
- Moyenne pondérée : News (60%) + Reddit (40%)
- Lissage par moyenne mobile 7 jours
- Z-score rolling 30 jours pour normaliser

### 4. Génération du signal
```
sentiment_ma > threshold  ET momentum > 0  →  BUY
sentiment_ma < -threshold ET momentum < 0  →  SELL
sinon                                       →  HOLD
```
Filtre anti-surtrading : pas de signal identique deux jours consécutifs.

### 5. Backtest
- **Stratégie** : long-only, entrée au close sur BUY, sortie sur SELL
- **Coûts** : 10 bps aller-retour (0.1% par trade)
- **Benchmark** : Buy & Hold naïf

**Métriques** : Sharpe Ratio, Max Drawdown, Win Rate, Calmar Ratio

---

## 💡 Pistes d'amélioration (pour aller plus loin)

1. **Sentiment intraday** : intégrer des données tick-by-tick
2. **Multi-actifs** : construire un portfolio basé sur les signaux croisés
3. **Machine Learning** : utiliser le sentiment comme feature dans un modèle XGBoost
4. **Options flow** : croiser avec les données d'options (put/call ratio)
5. **Événements** : détecter automatiquement les earnings, annonces Fed, etc.

---

## 📚 Références

- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [ProsusAI/finbert (HuggingFace)](https://huggingface.co/ProsusAI/finbert)
- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- [PRAW (Reddit API)](https://praw.readthedocs.io/)

---

*Projet réalisé dans le cadre d'une candidature en gestion de portefeuille.*  
*Les signaux générés sont à but pédagogique et ne constituent pas des conseils d'investissement.*
