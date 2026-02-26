import pandas as pd
import numpy as np
from typing import Dict


def compute_metrics(backtest_df: pd.DataFrame, risk_free_rate: float = 0.05) -> Dict:
    """
    Calcule les métriques de performance standard.

    Paramètres
    ----------
    backtest_df    : sortie de run_backtest()
    risk_free_rate : taux sans risque annualisé (défaut 5%)

    Retourne
    --------
    dict avec :
      total_return, sharpe, max_dd, win_rate, nb_trades,
      volatility, sharpe_bh, calmar
    """
    metrics = {}

    # Rendements stratégie
    ret = backtest_df['daily_ret_strategy'].dropna()
    ret_bh = backtest_df['daily_ret_bh'].dropna()
    cum = backtest_df['cumret_strategy'].dropna()

    # ── Performance totale ────────────────────────────────────────────────────
    metrics['total_return'] = float(cum.iloc[-1]) if not cum.empty else 0.0

    # ── Sharpe Ratio (annualisé) ──────────────────────────────────────────────
    rf_daily = risk_free_rate / 252
    excess_ret = ret - rf_daily
    metrics['sharpe'] = float(
        np.sqrt(252) * excess_ret.mean() / excess_ret.std()
        if excess_ret.std() > 1e-8 else 0.0
    )

    # Sharpe Buy & Hold
    excess_bh = ret_bh - rf_daily
    metrics['sharpe_bh'] = float(
        np.sqrt(252) * excess_bh.mean() / excess_bh.std()
        if excess_bh.std() > 1e-8 else 0.0
    )

    # ── Volatilité annualisée ─────────────────────────────────────────────────
    metrics['volatility'] = float(ret.std() * np.sqrt(252))

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    wealth = (1 + cum)
    rolling_max = wealth.cummax()
    drawdown = (wealth - rolling_max) / rolling_max
    metrics['max_dd'] = float(drawdown.min())

    # ── Calmar Ratio ─────────────────────────────────────────────────────────
    ann_ret = (1 + metrics['total_return']) ** (252 / max(len(ret), 1)) - 1
    metrics['calmar'] = float(ann_ret / abs(metrics['max_dd'])) if metrics['max_dd'] != 0 else 0.0

    # ── Win Rate ─────────────────────────────────────────────────────────────
    # Parmi les jours où on est en position, % de jours positifs
    in_position = backtest_df[backtest_df['position'] == 1]['daily_ret_strategy']
    metrics['win_rate'] = float((in_position > 0).mean()) if not in_position.empty else 0.5

    # ── Nombre de trades ─────────────────────────────────────────────────────
    trades = backtest_df['position'].diff().abs()
    metrics['nb_trades'] = int(trades.sum())

    return metrics
