# Pair Trading Backtest: Gold–Silver (or any two price series)
# -------------------------------------------------------------
# This notebook-like script builds a simple, transparent mean-reversion
# pair-trading framework using z-scores with:
# - Rolling cointegration proxy via log-spread with rolling hedge ratio
# - Volatility scaling between legs
# - Entry/exit on z-score bands
# - Basic transaction cost model
# - Performance stats + plots
#
# You can replace the synthetic data generator with your own CSVs by
# filling 'CSV_A' and 'CSV_B' paths. Expected CSV columns: ['date','close'].
#
# Notes:
# - Charts use matplotlib (no seaborn). Each chart is a single plot.
# - No internet calls here; purely local/simulated unless you load your own data.
#
# How to adapt quickly:
# - Set TICKER_A/TICKER_B (e.g., 'GOLD','SILVER' or 'Brent','WTI')
# - If you have local CSVs, set CSV_A/CSV_B to their file paths.
# - Tune params in the CONFIG block below.
#
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
import io, os, math, json
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Config:
    TICKER_A: str = "GOLD"
    TICKER_B: str = "SILVER"
    CSV_A: Optional[str] = None  # e.g., "/mnt/data/gold.csv"
    CSV_B: Optional[str] = None  # e.g., "/mnt/data/silver.csv"
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2025-10-01"
    FREQ: str = "B"                 # business days
    WINDOW_HEDGE: int = 60          # rolling window for hedge ratio
    WINDOW_ZSCORE: int = 60         # rolling z-score window
    ENTRY_Z: float = 1.5            # enter when |z| >= ENTRY_Z
    EXIT_Z: float = 0.5             # exit when |z| <= EXIT_Z
    COST_BPS_PER_LEG: float = 0.5   # roundtrip cost per leg (bps of notional)
    SLIPPAGE_BPS_PER_LEG: float = 0.5
    MAX_GROSS_LEVERAGE: float = 2.0 # cap |wA|+|wB|
    RISK_TARGET_DAILY: float = 0.01 # target gross exposure so daily vol ~1% (rough heuristic)
    MIN_DAYS_LIVE: int = 10         # ignore first days for warmup
    SEED: int = 42                  # for synthetic data reproducibility

cfg = Config()

# -----------------------------
# DATA LOADING / GENERATION
# -----------------------------
def load_or_simulate(cfg: Config) -> pd.DataFrame:
    if cfg.CSV_A and cfg.CSV_B and os.path.exists(cfg.CSV_A) and os.path.exists(cfg.CSV_B):
        a = pd.read_csv(cfg.CSV_A)
        b = pd.read_csv(cfg.CSV_B)
        a["date"] = pd.to_datetime(a["date"])
        b["date"] = pd.to_datetime(b["date"])
        df = (a[["date","close"]].rename(columns={"close":cfg.TICKER_A})
                .merge(b[["date","close"]].rename(columns={"close":cfg.TICKER_B}), on="date", how="inner"))
        df = df.sort_values("date").set_index("date")
        return df
    # Synthetic: correlated GBMs with occasional regime breaks
    np.random.seed(cfg.SEED)
    dates = pd.date_range(cfg.START_DATE, cfg.END_DATE, freq=cfg.FREQ)
    n = len(dates)
    # Base drifts/vols
    muA, muB = 0.06/252, 0.05/252
    volA, volB = 0.18/np.sqrt(252), 0.22/np.sqrt(252)
    rho = 0.75
    # Correlated shocks
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    epsA = z1
    epsB = rho*z1 + np.sqrt(1-rho**2)*z2
    # occasional regime in B to create spread opportunities
    regime = np.zeros(n)
    regime_idxs = np.random.choice(np.arange(100, n-100), size=6, replace=False)
    for idx in regime_idxs:
        regime[idx:idx+60] += np.random.choice([+0.001, -0.001])  # small drift shifts
    rA = muA + volA*epsA
    rB = muB + volB*epsB + regime
    pA = 100 * np.exp(np.cumsum(rA))
    pB = 80  * np.exp(np.cumsum(rB))
    df = pd.DataFrame({cfg.TICKER_A: pA, cfg.TICKER_B: pB}, index=dates)
    return df

prices = load_or_simulate(cfg)
prices = prices.dropna().copy()

# -----------------------------
# HELPERS
# -----------------------------
def rolling_hedge_ratio(logA: pd.Series, logB: pd.Series, window: int) -> pd.Series:
    # OLS beta of logA ~ alpha + beta*logB using rolling covariance/variance
    cov = logA.rolling(window).cov(logB)
    varB = logB.rolling(window).var()
    beta = cov / varB
    return beta

def compute_spread_and_z(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logA = np.log(prices[cfg.TICKER_A])
    logB = np.log(prices[cfg.TICKER_B])
    beta = rolling_hedge_ratio(logA, logB, cfg.WINDOW_HEDGE)
    spread = logA - beta*logB
    mean = spread.rolling(cfg.WINDOW_ZSCORE).mean()
    std  = spread.rolling(cfg.WINDOW_ZSCORE).std()
    z = (spread - mean) / std
    out = prices.copy()
    out["beta"] = beta
    out["spread"] = spread
    out["z"] = z
    return out.dropna()

def vol_scale_weights(retA: pd.Series, retB: pd.Series) -> Tuple[float,float]:
    # scale such that legs have similar risk contribution; use last 60d vol
    volA = retA.rolling(60).std().iloc[-1]
    volB = retB.rolling(60).std().iloc[-1]
    if not np.isfinite(volA) or not np.isfinite(volB) or volA==0 or volB==0:
        return 0.5, -0.5
    wA = 1/volA
    wB = -1/volB
    s = abs(wA)+abs(wB)
    return wA/s, wB/s

def backtest_pair(prices: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    df = compute_spread_and_z(prices, cfg)
    ret = prices.pct_change().reindex(df.index)
    retA = ret[cfg.TICKER_A]
    retB = ret[cfg.TICKER_B]

    # positions: - if z >= ENTRY_Z → short spread: short A, long B
    #            - if z <= -ENTRY_Z → long spread: long A, short B
    #            - exit if |z| <= EXIT_Z
    posA = np.zeros(len(df))
    posB = np.zeros(len(df))
    state = 0  # 0 flat, +1 long spread (long A short B), -1 short spread (short A long B)

    trade_entries = []
    trade_exits = []

    for i in range(len(df)):
        zi = df["z"].iloc[i]
        if state == 0:
            if zi >= cfg.ENTRY_Z:
                state = -1  # short spread
                trade_entries.append((df.index[i], "SHORT_SPREAD", zi))
            elif zi <= -cfg.ENTRY_Z:
                state = +1  # long spread
                trade_entries.append((df.index[i], "LONG_SPREAD", zi))
        else:
            if abs(zi) <= cfg.EXIT_Z:
                trade_exits.append((df.index[i], "EXIT", zi))
                state = 0

        posA[i] = 1.0 if state==+1 else (-1.0 if state==-1 else 0.0)
        posB[i] = -posA[i]  # opposite sign as baseline; will rescale by vol later

    # volatility scaling (static per day from rolling vols)
    wA_list, wB_list = [], []
    for i in range(len(df)):
        rA = retA.iloc[:i+1]
        rB = retB.iloc[:i+1]
        wA, wB = vol_scale_weights(rA, rB)
        gross = abs(wA)+abs(wB)
        if gross > 0:
            scale = min(cfg.MAX_GROSS_LEVERAGE / gross, 1.0)
            wA *= scale
            wB *= scale
        wA_list.append(wA)
        wB_list.append(wB)

    wA_arr = np.array(wA_list) * posA
    wB_arr = np.array(wB_list) * posB

    # PnL before costs
    pnl_raw = wA_arr * retA.iloc[:len(df)].values + wB_arr * retB.iloc[:len(df)].values

    # Transaction costs when positions change sign or go from/to zero
    # Approximate cost = (|ΔwA| + |ΔwB|) * (cost_bps + slippage_bps) / 1e4
    cost_bps = (cfg.COST_BPS_PER_LEG + cfg.SLIPPAGE_BPS_PER_LEG)
    dwA = np.abs(np.diff(np.insert(wA_arr, 0, 0.0)))
    dwB = np.abs(np.diff(np.insert(wB_arr, 0, 0.0)))
    costs = (dwA + dwB) * (cost_bps/1e4)

    pnl = pnl_raw - costs
    equity = (1 + pd.Series(pnl, index=df.index)).cumprod()

    # Stats
    daily_ret = pd.Series(pnl, index=df.index)
    ann_factor = 252
    cagr = equity.iloc[-1]**(ann_factor/len(equity)) - 1 if len(equity)>ann_factor else (equity.iloc[-1]-1)
    vol = daily_ret.std() * np.sqrt(ann_factor)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(ann_factor) if daily_ret.std()>0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    trades = max(len(trade_exits), len(trade_entries))

    stats = {
        "CAGR": float(cagr),
        "AnnVol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(dd),
        "Trades": int(trades),
        "Start": str(df.index[0].date()),
        "End": str(df.index[-1].date()),
        "EntryZ": cfg.ENTRY_Z,
        "ExitZ": cfg.EXIT_Z,
        "WindowHedge": cfg.WINDOW_HEDGE,
        "WindowZ": cfg.WINDOW_ZSCORE,
        "Cost_bps_per_leg_roundtrip": cfg.COST_BPS_PER_LEG,
        "Slippage_bps_per_leg_roundtrip": cfg.SLIPPAGE_BPS_PER_LEG,
    }

    res = df.copy()
    res["wA"] = wA_arr
    res["wB"] = wB_arr
    res["ret_pair"] = daily_ret.values
    res["equity"] = equity.values

    return res, stats, trade_entries, trade_exits

res, stats, entries, exits = backtest_pair(prices, cfg)

# -----------------------------
# OUTPUTS
# -----------------------------
from caas_jupyter_tools import display_dataframe_to_user

summary_df = pd.DataFrame([stats])
display_dataframe_to_user("Pair Backtest Stats (Synthetic GOLD–SILVER)", summary_df)

# Save signals/trades
signals_path = "/mnt/data/pair_signals.csv"
res_out = res[[cfg.TICKER_A, cfg.TICKER_B, "beta", "spread", "z", "wA", "wB", "ret_pair", "equity"]].copy()
res_out.to_csv(signals_path)

trades_df = pd.DataFrame(entries, columns=["time","type","z_entry"])
trades_df["exit_time"] = None
ex_iter = iter([e[0] for e in exits])
for i in range(len(trades_df)):
    try:
        trades_df.loc[i, "exit_time"] = next(ex_iter)
    except StopIteration:
        break

trades_path = "/mnt/data/pair_trades.csv"
trades_df.to_csv(trades_path, index=False)

# Plots
plt.figure(figsize=(10,4))
plt.plot(res.index, res["equity"])
plt.title("Equity Curve (Pair Strategy)")
plt.xlabel("Date")
plt.ylabel("Equity (rebased)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(res.index, res["z"])
plt.axhline(cfg.ENTRY_Z, linestyle="--")
plt.axhline(-cfg.ENTRY_Z, linestyle="--")
plt.axhline(cfg.EXIT_Z, linestyle=":")
plt.axhline(-cfg.EXIT_Z, linestyle=":")
plt.title("Z-Score of Spread (with Entry/Exit Bands)")
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.tight_layout()
plt.show()

# Save a self-contained Python script for your repo
script_code = f"""# pair_backtest.py
# Minimal, self-contained pair-trading backtest (z-score mean reversion)
# Usage: run as a script or import and replace the synthetic data with your own series.
{open(__file__).read() if '__file__' in globals() else "# (Notebook environment) Copy from the ChatGPT message."}
"""
script_path = "/mnt/data/pair_backtest.py"
with open(script_path, "w") as f:
    f.write(script_code)

signals_path, trades_path, script_path
