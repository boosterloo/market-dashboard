import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Cfg:
    window_beta: int = 60
    window_z: int = 60
    entry_z: float = 1.5
    exit_z: float = 0.5
    cost_bps_leg: float = 0.5
    slippage_bps_leg: float = 0.5
    max_gross_lev: float = 2.0
    seed: int = 42

cfg = Cfg()

# ----- data (synthetic demo). Vervang dit blok door je CSV load -----
rng = np.random.default_rng(cfg.seed)
dates = pd.bdate_range("2018-01-01", "2025-10-01")
n = len(dates)
muA, muB = 0.06/252, 0.05/252
volA, volB = 0.18/np.sqrt(252), 0.22/np.sqrt(252)
rho = 0.75
z1 = rng.standard_normal(n); z2 = rng.standard_normal(n)
epsA = z1; epsB = rho*z1 + np.sqrt(1-rho**2)*z2
regime = np.zeros(n)
for idx in rng.choice(np.arange(100, n-100), 6, replace=False):
    regime[idx:idx+60] += rng.choice([+0.001, -0.001])
rA = muA + volA*epsA
rB = muB + volB*epsB + regime
A = 100*np.exp(np.cumsum(rA)); B = 80*np.exp(np.cumsum(rB))
prices = pd.DataFrame({"A": A, "B": B}, index=dates)

# ----- helpers -----
def rolling_beta(x, y, w):
    cov = x.rolling(w).cov(y)
    var = y.rolling(w).var()
    return cov/var

def vol_scale(retA, retB):
    vA = retA.rolling(60).std().iloc[-1]; vB = retB.rolling(60).std().iloc[-1]
    if not np.isfinite(vA) or not np.isfinite(vB) or vA==0 or vB==0: return 0.5, -0.5
    wA, wB = 1/vA, -1/vB
    s = abs(wA)+abs(wB); wA/=s; wB/=s
    gross = abs(wA)+abs(wB)
    if gross>cfg.max_gross_lev:
        f = cfg.max_gross_lev/gross; wA*=f; wB*=f
    return wA, wB

# ----- core calcs -----
logA, logB = np.log(prices["A"]), np.log(prices["B"])
beta = rolling_beta(logA, logB, cfg.window_beta)
spread = logA - beta*logB
mu, sd = spread.rolling(cfg.window_z).mean(), spread.rolling(cfg.window_z).std()
z = (spread-mu)/sd
df = prices.copy()
df["beta"], df["spread"], df["z"] = beta, spread, z
df = df.dropna()
ret = prices.pct_change().reindex(df.index)
retA, retB = ret["A"], ret["B"]

state = 0
posA = np.zeros(len(df)); posB = np.zeros(len(df))
entries, exits = [], []
for i, (ts, zi) in enumerate(zip(df.index, df["z"].values)):
    if state==0:
        if zi>=cfg.entry_z:  state=-1; entries.append((ts,"SHORT",zi))
        elif zi<=-cfg.entry_z: state=+1; entries.append((ts,"LONG",zi))
    else:
        if abs(zi)<=cfg.exit_z: exits.append((ts,"EXIT",zi)); state=0
    posA[i] = 1.0 if state==+1 else (-1.0 if state==-1 else 0.0)
    posB[i] = -posA[i]

wA_list, wB_list = [], []
for i in range(len(df)):
    wA, wB = vol_scale(retA.iloc[:i+1], retB.iloc[:i+1])
    wA_list.append(wA*posA[i]); wB_list.append(wB*posB[i])
wA, wB = np.array(wA_list), np.array(wB_list)

pnl_raw = wA*retA.values + wB*retB.values
cost_bps = (cfg.cost_bps_leg + cfg.slippage_bps_leg)
dwA = np.abs(np.diff(np.insert(wA,0,0.0))); dwB = np.abs(np.diff(np.insert(wB,0,0.0)))
costs = (dwA+dwB)*(cost_bps/1e4)
pnl = pnl_raw - costs
equity = (1+pd.Series(pnl, index=df.index)).cumprod()

# ----- stats -----
ann = 252
r = pd.Series(pnl, index=df.index)
cagr = equity.iloc[-1]**(ann/len(equity))-1 if len(equity)>ann else equity.iloc[-1]-1
vol = r.std()*np.sqrt(ann)
sharpe = (r.mean()/r.std())*np.sqrt(ann) if r.std()>0 else np.nan
mdd = (equity/equity.cummax() - 1).min()
print(f"CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | AnnVol: {vol:.2%} | MaxDD: {mdd:.2%} | Trades: {max(len(entries),len(exits))}")

# ----- exports -----
out = df[["A","B","beta","spread","z"]].copy()
out["wA"], out["wB"], out["ret_pair"], out["equity"] = wA, wB, r.values, equity.values
out.to_csv("pair_signals.csv")
pd.DataFrame(entries, columns=["time","side","z"]).to_csv("pair_entries.csv", index=False)
pd.DataFrame(exits, columns=["time","side","z"]).to_csv("pair_exits.csv", index=False)

# ----- plots -----
plt.figure(figsize=(10,4)); plt.plot(equity.index, equity.values)
plt.title("Equity Curve"); plt.xlabel("Date"); plt.ylabel("Equity"); plt.tight_layout(); plt.savefig("equity.png")

plt.figure(figsize=(10,4)); plt.plot(df.index, df["z"].values)
for y,ls in [(cfg.entry_z,"--"),(-cfg.entry_z,"--"),(cfg.exit_z,":"),(-cfg.exit_z,":")]:
    plt.axhline(y=y, linestyle=ls)
plt.title("Z-Score (bands)"); plt.xlabel("Date"); plt.ylabel("Z"); plt.tight_layout(); plt.savefig("zscore.png")
