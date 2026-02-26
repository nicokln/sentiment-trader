import pandas as pd
import numpy as np
from typing import Tuple, Dict


def run_backtest(
    signal_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    transaction_cost: float = 0.001,  # 10 bps aller-retour
    initial_capital: float = 10_000.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest simple : long-only, on entre sur signal BUY, on sort sur SELL ou fin.

    Paramètres
    ----------
    signal_df         : sortie de generate_signal()
    prices_df         : OHLCV
    transaction_cost  : coût par transaction (proportion du prix, par sens)
    initial_capital   : capital initial en $

    Retourne
    --------
    (backtest_df, perf_dict)
    backtest_df : DataFrame avec colonnes position, daily_ret, cumret_strategy, cumret_bh
    perf_dict   : dictionnaire de métriques résumées
    """
    # Alignement sur les jours de bourse communs
    common_idx = signal_df.index.intersection(prices_df.index)
    sig = signal_df.loc[common_idx, 'signal']
    prices = prices_df.loc[common_idx, 'Close']

    # Rendements journaliers
    daily_ret = prices.pct_change().fillna(0)

    # Position : 1 = long, 0 = pas en position
    # On rentre au close du jour du signal, on sort le jour suivant
    position = pd.Series(0, index=common_idx)
    in_position = False

    for i, (date, signal) in enumerate(sig.items()):
        if signal == 'BUY' and not in_position:
            in_position = True
        elif signal == 'SELL' and in_position:
            in_position = False
        position.loc[date] = 1 if in_position else 0

    # Décalage : on applique la position du jour J sur le rendement J+1
    position_shifted = position.shift(1).fillna(0)

    # Rendement stratégie net des coûts de transaction
    trades = position_shifted.diff().abs()  # 1 quand changement de position
    strategy_ret = position_shifted * daily_ret - trades * transaction_cost

    # Performance cumulée
    cumret_strategy = (1 + strategy_ret).cumprod() - 1
    cumret_bh = (1 + daily_ret).cumprod() - 1

    # Capital en valeur absolue
    portfolio_value = initial_capital * (1 + cumret_strategy)

    # Nombre de trades
    nb_trades = int(trades.sum())

    backtest_df = pd.DataFrame({
        'position': position_shifted,
        'daily_ret_bh': daily_ret,
        'daily_ret_strategy': strategy_ret,
        'cumret_strategy': cumret_strategy,
        'cumret_bh': cumret_bh,
        'portfolio_value': portfolio_value
    })

    perf = {
        'total_return_strategy': float(cumret_strategy.iloc[-1]),
        'total_return_bh': float(cumret_bh.iloc[-1]),
        'nb_trades': nb_trades,
    }

    return backtest_df, perf
