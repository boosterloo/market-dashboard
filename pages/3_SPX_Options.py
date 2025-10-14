# ─────────────────────────────────────────────────────────────
# pages/3_SPX_Options.py — stabiele versie (deel A)
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
from google.oauth2 import service_account

# ------------------------------------------
# Eigen cumulatieve normaal (geen scipy)
# ------------------------------------------
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ------------------------------------------
# Black–Scholes Delta
# ------------------------------------------
def bs_delta(S, K, iv, T, r, q, is_call):
    if np.isnan(S) or np.isnan(K) or np.isnan(iv) or T <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    Nd1 = norm_cdf(d1)
    disc_q = math.exp(-q * T)
    return disc_q * Nd1 if is_call else disc_q * (Nd1 - 1.0)

# ------------------------------------------
# Basic payoffs en marges
# ------------------------------------------
def strangle_payoff_at_expiry(S, Kp, Kc, credit_pts, multiplier=100):
    return (credit_pts - max(Kp - S, 0) - max(S - Kc, 0)) * multiplier

def span_like_margin(S, Kp, Kc, credit_pts, down=0.15, up=0.10, multiplier=100):
    S_down, S_up = S*(1-down), S*(1+up)
    loss_down = (max(Kp - S_down, 0) - credit_pts) * multiplier
    loss_up = (max(S_up - Kc, 0) - credit_pts) * multiplier
    return max(0.0, loss_down, loss_up)

# ------------------------------------------
# BigQuery setup
# ------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bq_client():
    sa = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa)
    return bigquery.Client(project=sa["project_id"], credentials=creds)

bq = get_bq_client()

def run_query(sql: str, params=None):
    job_cfg = None
    if params:
        job_cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", str(v)) for k, v in params.items()
            ]
        )
    df = bq.query(sql, job_config=job_cfg).to_dataframe()
    return df

# ------------------------------------------
# Utilities
# ------------------------------------------
def smooth_series(y: pd.Series, window: int = 3):
    if len(y) < 3:
        return y
    return y.rolling(window, center=True, min_periods=1).median()

def pick_closest(options, value):
    if not options:
        return None
    return min(options, key=lambda x: abs(float(x) - float(value)))

# ─────────────────────────────────────────────────────────────
# pages/3_SPX_Options.py — stabiele versie (deel B)
# ─────────────────────────────────────────────────────────────

# Page config + titel
st.set_page_config(page_title="🧰 SPX Options — Skew, Delta & PPD", layout="wide")
st.title("🧰 SPX Options — Skew, Delta & PPD")

# Volledig gekwalificeerde BigQuery view
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
VIEW = f"{PROJECT_ID}.marketdata.spx_options_enriched_v"

# ---------------------
# Sidebar instellingen
# ---------------------
with st.sidebar:
    st.header("⚙️ Instellingen")
    center_mode = st.radio(
        "Strike-centrering", ["Rounded (aanbevolen)", "ATM (live underlying)"], index=0
    )
    round_base = st.select_slider("Rond strikes op", options=[25, 50, 100], value=25)
    max_pts = st.slider("Afstand tot (gecentreerde) strike (± punten)", 50, 1000, 400, 50)
    dte_pref = st.selectbox("DTE-selectie voor skew", ["Nearest", "0–7", "8–21", "22–45", "46–90", "90+"])
    r_input = st.number_input("Risicovrije rente r (p.j.)", value=0.00, step=0.25, format="%.2f")
    q_input = st.number_input("Dividend/Index carry q (p.j.)", value=0.00, step=0.25, format="%.2f")
    st.caption("Tip: **Rounded + ±400–600** geeft stabiele skew-curves. ATM kan intraday verschuiven.")

# ---------------------
# Data laden (laatste snapshot)
# ---------------------
@st.cache_data(ttl=600, show_spinner=True)
def load_latest_snapshot(view_fqn: str) -> pd.DataFrame:
    sql = f"""
    WITH last AS (SELECT MAX(snapshot_date) AS snapshot_date FROM `{view_fqn}`)
    SELECT contract_symbol, type, expiration, days_to_exp, strike, underlying_price,
           last_price, bid, ask, mid_price, implied_volatility, open_interest, volume,
           vix, snapshot_date
    FROM `{view_fqn}`
    WHERE snapshot_date = (SELECT snapshot_date FROM last)
    """
    return run_query(sql)

df = load_latest_snapshot(VIEW)
if df.empty:
    st.warning("Geen SPX-optiedata gevonden in de view.")
    st.stop()

# Clean & types
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.date
for c in ["strike","underlying_price","implied_volatility","days_to_exp","open_interest","volume","last_price","bid","ask","mid_price"]:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["strike","underlying_price","implied_volatility","days_to_exp"]).copy()
df["type"] = df["type"].astype(str).str.lower()

# ---------------------
# Centrering: points-to-strike
# ---------------------
S_now = float(pd.to_numeric(df["underlying_price"], errors="coerce").dropna().median())
center = round_base * round(S_now / round_base) if center_mode.startswith("Rounded") else S_now
df["pts_to_strike"] = df["strike"] - center
df = df[df["pts_to_strike"].between(-max_pts, max_pts)].copy()

if df.empty:
    st.info("Geen rijen binnen de ingestelde ± afstand.")
    st.stop()

# ---------------------
# DTE selectie (voor skew sectie)
# ---------------------
if dte_pref == "Nearest":
    # kies de exact kleinste DTE (meest nabij) binnen de gefilterde set
    target_dte = float(df["days_to_exp"].min())
    skew_df = df.loc[df["days_to_exp"] == target_dte].copy()
else:
    lo, hi = {"0–7":(0,7), "8–21":(8,21), "22–45":(22,45), "46–90":(46,90), "90+":(90, 10_000)}[dte_pref]
    skew_df = df.loc[df["days_to_exp"].between(lo, hi)].copy()

if skew_df.empty:
    st.info("Geen rijen in de gekozen DTE-bucket.")
    st.stop()

# ---------------------
# Basis KPI’s
# ---------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Underlying (S₀)", f"{S_now:,.0f}")
c2.metric("Center", f"{center:,.0f}")
_dte_med = pd.to_numeric(skew_df["days_to_exp"], errors="coerce").median()
c3.metric("DTE (skew)", f"{max(0, int(_dte_med))} d" if not np.isnan(_dte_med) else "—")
c4.metric("Rijen (skew)", f"{len(skew_df):,}")

st.markdown("---")

# ---------------------
# Voorraad van expiraties/strikes (defaults voor latere secties)
# ---------------------
exps_all = sorted(pd.Series(df["expiration"].unique()).dropna().tolist())
strikes_all = sorted([float(x) for x in pd.to_numeric(df["strike"], errors="coerce").dropna().unique().tolist()])

# veilige defaults
target_exp = date.today() + timedelta(days=14)
def _pick_first_on_or_after(opts, tgt):
    try:
        opts = [pd.to_datetime(d).date() for d in opts if pd.notna(d)]
        tgt = pd.to_datetime(tgt).date()
        aft = [d for d in opts if d >= tgt]
        return aft[0] if aft else (opts[-1] if opts else None)
    except Exception:
        return opts[0] if opts else None

def _pick_closest(options, target):
    if not options: return None
    return min(options, key=lambda x: abs(float(x) - float(target)))

default_series_exp = _pick_first_on_or_after(exps_all, target_exp) or (exps_all[0] if exps_all else None)
default_series_strike = _pick_closest(
    strikes_all,
    (S_now - 300.0) if (not np.isnan(S_now) and not np.isinf(S_now)) else (strikes_all[len(strikes_all)//2] if strikes_all else 6000.0)
)

# ---------------------
# Plot-config
# ---------------------
PLOTLY_CONFIG = {
    "scrollZoom": True, "doubleClick": "reset", "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}
# ─────────────────────────────────────────────────────────────
# pages/3_SPX_Options.py — stabiele versie (deel C)
# ─────────────────────────────────────────────────────────────

# ---------------------
# Delta berekening (vectorized) voor skew/gamma-stukken
# ---------------------
def bs_delta_vectorized(S, K, IV, T_days, r, q, is_call):
    """
    Vectorized Black–Scholes delta (continuous r,q). T_days in dagen.
    S,K,IV,T_days en is_call kunnen arrays/Series zijn; r en q scalars.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    IV = np.asarray(IV, dtype=float)
    T = np.asarray(T_days, dtype=float) / 365.0
    is_call = np.asarray(is_call, dtype=bool)

    n = len(S)
    r_arr = np.full(n, float(r), dtype=float)
    q_arr = np.full(n, float(q), dtype=float)

    eps = 1e-12
    sigma = np.maximum(IV, eps)
    sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(np.maximum(S, eps) / np.maximum(K, eps)) + (r_arr - q_arr + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    disc = np.exp(-q_arr * T)
    # call:  e^{-qT} N(d1),  put: -e^{-qT} N(-d1)
    return np.where(is_call, disc * norm_cdf(d1), -disc * norm_cdf(-d1)).astype(float)

# ---------------------
# 1) Skew — IV & Delta vs points-to-strike
# ---------------------
st.subheader("Skew — IV & Δ vs afstand (points to strike)")

# Delta voor skew_df
skew_df = skew_df.copy()
skew_df["is_call"] = skew_df["type"].eq("call")
skew_df["delta"] = bs_delta_vectorized(
    S=skew_df["underlying_price"].to_numpy(),
    K=skew_df["strike"].to_numpy(),
    IV=skew_df["implied_volatility"].to_numpy(),
    T_days=skew_df["days_to_exp"].to_numpy(),
    r=r_input,
    q=q_input,
    is_call=skew_df["is_call"].to_numpy()
)

fig_skew = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("Implied Volatility (IV) vs Points to strike",
                                         "Delta (Δ) vs Points to strike"))

for side in ["call", "put"]:
    sub = skew_df[skew_df["type"] == side]
    if sub.empty:
        continue
    fig_skew.add_trace(
        go.Scatter(
            x=sub["pts_to_strike"], y=sub["implied_volatility"], mode="markers",
            name=f"IV {side}", hovertemplate="pts: %{x:.0f}<br>IV: %{y:.1%}<extra></extra>"
        ),
        row=1, col=1
    )
    fig_skew.add_trace(
        go.Scatter(
            x=sub["pts_to_strike"], y=sub["delta"], mode="markers",
            name=f"Δ {side}", hovertemplate="pts: %{x:.0f}<br>Δ: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

fig_skew.update_xaxes(title_text="Points to strike (K − center)", row=2, col=1)
fig_skew.update_yaxes(title_text="IV", tickformat=".0%", row=1, col=1)
fig_skew.update_yaxes(title_text="Δ", row=2, col=1)
fig_skew.update_layout(height=680, showlegend=True, margin=dict(t=60, b=40, l=40, r=20))
st.plotly_chart(fig_skew, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")

# ---------------------
# 2) Term Structure & IV Smile (laatste snapshot)
# ---------------------
st.subheader("Term Structure — mediane IV per points-band")

# Points-bands rond center
bins = [-10_000, -200, -50, 50, 200, 10_000]
labels = ["Put far (≤−200)", "Put near (−200..−50)", "Near ATM (−50..50)", "Call near (50..200)", "Call far (≥200)"]
df["pts_band"] = pd.cut(df["pts_to_strike"], bins=bins, labels=labels)

ts = (df.groupby(["days_to_exp", "pts_band"], as_index=False)["implied_volatility"]
        .median().dropna())
fig_ts = go.Figure()
for band in labels:
    sub = ts.loc[ts["pts_band"] == band]
    if sub.empty:
        continue
    fig_ts.add_trace(
        go.Scatter(
            x=sub["days_to_exp"], y=sub["implied_volatility"], mode="lines+markers", name=band,
            hovertemplate="DTE: %{x:.0f}d<br>IV: %{y:.1%}<extra></extra>"
        )
    )
fig_ts.update_layout(height=420, xaxis_title="DTE (dagen)", yaxis_title="Mediane IV", yaxis_tickformat=".0%")
st.plotly_chart(fig_ts, use_container_width=True, config=PLOTLY_CONFIG)

st.subheader("IV Smile (laatste snapshot)")
exps_for_smile = sorted(pd.Series(df["expiration"].unique()).dropna().tolist())
exp_for_smile = st.selectbox("Expiratie voor IV Smile", options=exps_for_smile or [None], index=0)
sm = df[df["expiration"] == exp_for_smile].copy() if exp_for_smile else pd.DataFrame()
if sm.empty:
    st.info("Geen data voor gekozen expiratie.")
else:
    sm = sm.groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
    if len(sm) >= 5:
        lo, hi = sm["implied_volatility"].quantile([0.02, 0.98])
        sm["implied_volatility"] = sm["implied_volatility"].clip(lower=lo, upper=hi)
    fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV"))
    fig_sm.update_layout(height=420, xaxis_title="Strike", yaxis_title="Implied Volatility", yaxis_tickformat=".0%")
    st.plotly_chart(fig_sm, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")

# ---------------------
# 3) Put/Call-ratio en Vol & Risk
# ---------------------
st.subheader("Put/Call-ratio per expiratie")
p = (df.groupby(["expiration", "type"], as_index=False)
       .agg(vol=("volume", "sum"), oi=("open_interest", "sum")))
if p.empty:
    st.info("Geen data voor PCR.")
else:
    pv = (p.pivot_table(index="expiration", columns="type", values=["vol", "oi"], aggfunc="sum")
            .sort_index().sort_index(axis=1)).fillna(0.0)
    pv.columns = [f"{a}_{b}" for a, b in pv.columns.to_flat_index()]
    for col in ["vol_put", "vol_call", "oi_put", "oi_call"]:
        if col not in pv.columns:
            pv[col] = 0.0
    pv["PCR_vol"] = pv["vol_put"] / pv["vol_call"].replace(0, np.nan)
    pv["PCR_oi"] = pv["oi_put"] / pv["oi_call"].replace(0, np.nan)
    pv = pv.replace([np.inf, -np.inf], np.nan).dropna(subset=["PCR_vol", "PCR_oi"], how="all")
    if pv.empty:
        st.info("Niet genoeg data (alleen puts of alleen calls in de selectie).")
    else:
        fig_pcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("PCR op Volume", "PCR op Open Interest"))
        fig_pcr.add_trace(go.Bar(x=pv.index.astype(str), y=pv["PCR_vol"], name="PCR (Vol)"), row=1, col=1)
        fig_pcr.add_trace(go.Bar(x=pv.index.astype(str), y=pv["PCR_oi"], name="PCR (OI)"), row=2, col=1)
        fig_pcr.update_layout(height=520, title_text="Put/Call-ratio per Expiratie")
        st.plotly_chart(fig_pcr, use_container_width=True, config=PLOTLY_CONFIG)

st.subheader("📊 Vol & Risk (ATM-IV, HV, VRP, IV-Rank, Expected Move)")
# Underlying dagreeks uit laatste snapshot (median per dag)
u_daily = (df.assign(day=df["snapshot_date"].dt.date)
             .groupby("day", as_index=False)
             .agg(close=("underlying_price", "median")))
u_daily["ret"] = pd.to_numeric(u_daily["close"], errors="coerce").pct_change()

def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))

hv20 = annualize_std(u_daily["ret"].tail(21).dropna())

near_atm = df[(df["days_to_exp"].between(20, 40)) & ((df["strike"] / df["underlying_price"] - 1.0).abs() <= 0.01)]
iv_atm = float(near_atm["implied_volatility"].median()) if not near_atm.empty else float(df["implied_volatility"].median())

iv_hist = (df[(df["days_to_exp"].between(20, 40)) & ((df["strike"] / df["underlying_price"] - 1.0).abs() <= 0.01)]
           .groupby(df["snapshot_date"].dt.date, as_index=False)["implied_volatility"].median()
           .rename(columns={"implied_volatility": "iv"}))
iv_1y = iv_hist["iv"].tail(252) if not iv_hist.empty else pd.Series(dtype=float)
iv_rank = float((iv_1y <= iv_1y.iloc[-1]).mean()) if len(iv_1y) >= 2 else np.nan

dte_selected = int(pd.to_numeric(df["days_to_exp"], errors="coerce").median()) if not df.empty else 30
em_sigma = (S_now * iv_atm * math.sqrt(max(dte_selected, 1) / 365.0)) if (not np.isnan(S_now) and not np.isnan(iv_atm)) else np.nan

cv1, cv2, cv3, cv4, cv5 = st.columns(5)
cv1.metric("ATM-IV (~30D)", f"{iv_atm:.2%}" if not np.isnan(iv_atm) else "—")
cv2.metric("HV20", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
cv3.metric("VRP (IV−HV)", f"{(iv_atm - hv20):.2%}" if (not np.isnan(iv_atm) and not np.isnan(hv20)) else "—")
cv4.metric("IV-Rank (1y)", f"{iv_rank*100:.0f}%" if not np.isnan(iv_rank) else "—")
cv5.metric("Expected Move (σ)", f"±{em_sigma:,.0f} pts ({em_sigma/S_now:.2%})" if (not np.isnan(em_sigma) and S_now > 0) else "—")

st.markdown("---")

# ---------------------
# 4) Strangle Helper (σ- of Δ-doel)
# ---------------------
st.subheader("🧠 Strangle Helper (σ- of Δ-doel)")

# UI
cm1, cm2, cm3, cm4 = st.columns([1.2, 1, 1, 1])
with cm1: str_sel_mode = st.radio("Selectiemodus", ["σ-doel", "Δ-doel"], index=0)
with cm2: sigma_target = st.slider("σ-doel per zijde", 0.5, 2.5, 1.0, step=0.1)
with cm3: delta_target = st.slider("Δ-doel (absoluut)", 0.05, 0.30, 0.15, step=0.01)
with cm4: price_source = st.radio("Prijsbron", ["mid_price", "last_price"], index=0, horizontal=True)

exps_all = sorted(pd.Series(df["expiration"].unique()).dropna().tolist())
if not exps_all:
    st.info("Geen expiraties beschikbaar op dit snapshot.")
else:
    exp_for_str = st.selectbox("Expiratie voor strangle", options=exps_all, index=0)
    df_str = df[df["expiration"] == exp_for_str].copy()
    df_str["mny"] = df_str["strike"] / df_str["underlying_price"] - 1.0
    df_str = df_str[((df_str["open_interest"].fillna(0) >= 1) | (df_str["volume"].fillna(0) >= 1))]
    if df_str.empty:
        st.info("Geen liquide rijen voor deze expiratie.")
    else:
        iv_atm_exp = float(df_str.loc[(df_str["days_to_exp"].between(20, 60)) & (df_str["mny"].abs() <= 0.01), "implied_volatility"].median())
        dte_exp = int(pd.to_numeric(df_str["days_to_exp"], errors="coerce").median())
        T_years = max(dte_exp, 1) / 365.0
        sigma_pts = S_now * iv_atm_exp * math.sqrt(T_years)

        smile_map = (df_str.groupby(["type", "strike"], as_index=False)["implied_volatility"].median()
                        .set_index(["type", "strike"])["implied_volatility"].to_dict())
        use_smile_iv = st.checkbox("Gebruik strike-IV (smile) voor Δ", value=False)

        def get_iv(side: str, K: float) -> float:
            if use_smile_iv:
                v = smile_map.get((side, K), np.nan)
                if not np.isnan(v):
                    return float(v)
            return float(iv_atm_exp)

        def nearest_strike(side: str, target_price: float) -> float:
            s_list = sorted(df_str[df_str["type"] == side]["strike"].unique().tolist())
            if not s_list:
                return np.nan
            return float(min(s_list, key=lambda x: abs(float(x) - float(target_price))))

        def pick_by_sigma():
            if np.isnan(sigma_pts):
                return np.nan, np.nan
            return (nearest_strike("put", S_now - sigma_target * sigma_pts),
                    nearest_strike("call", S_now + sigma_target * sigma_pts))

        def pick_by_delta():
            puts = sorted(df_str[df_str["type"] == "put"]["strike"].unique().tolist())
            calls = sorted(df_str[df_str["type"] == "call"]["strike"].unique().tolist())
            best_p, best_c, err_p, err_c = np.nan, np.nan, 1e9, 1e9
            # vectorized delta per K
            for K in puts:
                d = bs_delta_vectorized([S_now], [K], [get_iv("put", K)], [dte_exp], r_input, q_input, [False])[0]
                e = abs(abs(d) - delta_target)
                if not np.isnan(d) and e < err_p:
                    best_p, err_p = K, e
            for K in calls:
                d = bs_delta_vectorized([S_now], [K], [get_iv("call", K)], [dte_exp], r_input, q_input, [True])[0]
                e = abs(d - delta_target)
                if not np.isnan(d) and e < err_c:
                    best_c, err_c = K, e
            return float(best_p), float(best_c)

        target_put, target_call = (pick_by_sigma() if str_sel_mode.startswith("σ") else pick_by_delta())

        def _px(typ, K):
            row = df_str[(df_str["type"] == typ) & (df_str["strike"] == K)]
            return float(pd.to_numeric(row[price_source], errors="coerce").median()) if not row.empty else np.nan

        put_px, call_px = _px("put", target_put), _px("call", target_call)
        total_credit = (put_px + call_px) if (not np.isnan(put_px) and not np.isnan(call_px)) else np.nan

        # σ-afstand / ~P(touch)
        def sigma_distance(K: float) -> float:
            return abs(K - S_now) / sigma_pts if (sigma_pts and not np.isnan(sigma_pts)) else np.nan

        sd_put, sd_call = sigma_distance(target_put), sigma_distance(target_call)

        def p_itm_at_exp(sd: float) -> float:
            # ≈ P(ITM at expiry) proxy via N̄(sd)
            return (1.0 - norm_cdf(sd)) if not np.isnan(sd) else np.nan

        p_touch_put = min(1.0, 2.0 * p_itm_at_exp(sd_put)) if not np.isnan(sd_put) else np.nan
        p_touch_call = min(1.0, 2.0 * p_itm_at_exp(sd_call)) if not np.isnan(sd_call) else np.nan
        p_both_touch = min(1.0, (p_touch_put or 0.0) + (p_touch_call or 0.0))

        ppd_total_pts = float(total_credit / max(dte_exp, 1)) if (not np.isnan(total_credit) and not np.isnan(dte_exp)) else np.nan

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Expiratie", str(exp_for_str))
        k2.metric("DTE", f"{dte_exp:.0f}")
        k3.metric("Strikes", (f"P {target_put:.0f} / C {target_call:.0f}") if not (np.isnan(target_put) or np.isnan(target_call)) else "—")
        k4.metric("Credit (pts)", f"{total_credit:,.2f}" if not np.isnan(total_credit) else "—")
        k5.metric("PPD (pts, 1x)", f"{ppd_total_pts:,.2f}" if not np.isnan(ppd_total_pts) else "—")
        k6.metric("~P(touch) max", f"{p_both_touch*100:.0f}%" if not np.isnan(p_both_touch) else "—")

        st.markdown("---")

        # ---------------------
        # 5) Margin & Payoff
        # ---------------------
        st.subheader("💳 Margin & Payoff")
        ready = (not any(np.isnan(x) for x in [S_now, target_put, target_call])) and (not np.isnan(total_credit))
        if not ready:
            st.info("Kies eerst een geldige strangle (σ of Δ).")
        else:
            sm1, sm2, sm3, sm4 = st.columns([1.1, 1, 1, 1])
            with sm1: margin_model = st.radio("Margin model", ["SPAN-like stress", "Reg-T approx"], index=0)
            with sm2: down_shock = st.slider("Down shock (%)", 5, 30, 15, step=1)
            with sm3: up_shock = st.slider("Up shock (%)", 5, 30, 10, step=1)
            with sm4: multiplier = st.number_input("Contract multiplier", min_value=10, max_value=250, value=100, step=10)

            if margin_model.startswith("SPAN"):
                est_margin = span_like_margin(S_now, float(target_put), float(target_call), float(total_credit),
                                              down=down_shock/100.0, up=up_shock/100.0, multiplier=multiplier)
            else:
                # eenvoudige Reg-T benadering
                otm_call, otm_put = max(float(target_call) - S_now, 0.0), max(S_now - float(target_put), 0.0)
                base_call = max(0.20 * S_now - otm_call, 0.10 * S_now)
                base_put = max(0.20 * S_now - otm_put, 0.10 * S_now)
                call_px_pts = float(call_px) if not np.isnan(call_px) else 0.0
                put_px_pts = float(put_px) if not np.isnan(put_px) else 0.0
                req_call = (call_px_pts + base_call) * multiplier
                req_put = (put_px_pts + base_put) * multiplier
                worst_leg = max(req_call, req_put)
                other_leg = put_px_pts if worst_leg == req_call else call_px_pts
                est_margin = float(worst_leg + other_leg * multiplier)

            ref_budget = st.number_input("Referentie-budget voor #contracts", min_value=1000, value=10000, step=1000)
            n_contracts = int(np.floor(ref_budget / est_margin)) if est_margin > 0 else 0
            tot_credit_cash = (float(total_credit) if not np.isnan(total_credit) else 0.0) * multiplier
            credit_per_margin = (tot_credit_cash / est_margin) if est_margin > 0 else np.nan
            ppd_per_margin = ((ppd_total_pts * multiplier) / est_margin) if (est_margin > 0 and not np.isnan(ppd_total_pts)) else np.nan

            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Est. margin (1x)", f"{est_margin:,.0f}")
            mm2.metric("# Contracts @budget", f"{n_contracts:,}")
            mm3.metric("Credit (1x)", f"{tot_credit_cash:,.0f}")
            mm4.metric("Credit / Margin", f"{credit_per_margin:.2f}" if not np.isnan(credit_per_margin) else "—")

            show_payoff = st.checkbox("Toon payoff (1x)", value=True)
            if show_payoff:
                rng = 0.25
                S_grid = np.linspace(S_now*(1-rng), S_now*(1+rng), 400)
                pnl_grid = [strangle_payoff_at_expiry(s, float(target_put), float(target_call), float(total_credit), multiplier=multiplier) for s in S_grid]
                be_low = float(target_put) - float(total_credit)
                be_high = float(target_call) + float(total_credit)
                fig_pay = go.Figure()
                fig_pay.add_trace(go.Scatter(x=S_grid, y=pnl_grid, mode="lines", name="PNL @ expiry (1x)"))
                fig_pay.add_hline(y=0, line=dict(dash="dot"))
                fig_pay.add_vline(x=be_low, line=dict(dash="dot"), annotation_text=f"BE low ≈ {be_low:.0f}")
                fig_pay.add_vline(x=be_high, line=dict(dash="dot"), annotation_text=f"BE high ≈ {be_high:.0f}")
                fig_pay.update_layout(height=420, title=f"Payoff @ Expiry (P {target_put:.0f} / C {target_call:.0f} | credit {total_credit:.2f} pts)",
                                      xaxis_title="S (onderliggende)", yaxis_title=f"PNL (1x, multiplier {multiplier})")
                st.plotly_chart(fig_pay, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("---")

        # ---------------------
        # 6) Roll-simulator
        # ---------------------
        st.subheader("🔄 Roll-simulator")
        future_exps = [e for e in exps_all if e > exp_for_str]
        if not future_exps:
            st.info("Geen latere expiraties beschikbaar binnen je filters.")
        elif not ready:
            st.info("Selecteer eerst een strangle in de Strangle Helper.")
        else:
            rr1, rr2 = st.columns([1.2, 1])
            with rr1: roll_mode = st.radio("Rol-methode", ["σ-doel", "Δ-doel"], index=0, horizontal=True)
            with rr2: new_exp = st.selectbox("Naar welke expiratie rollen?", options=future_exps, index=0)

            df_new = df[df["expiration"] == new_exp].copy()
            df_new["mny"] = df_new["strike"] / df_new["underlying_price"] - 1.0
            df_new = df_new[((df_new["open_interest"].fillna(0) >= 1) | (df_new["volume"].fillna(0) >= 1))]
            if df_new.empty:
                st.info("Geen data voor de gekozen nieuwe expiratie.")
            else:
                dte_new = int(pd.to_numeric(df_new["days_to_exp"], errors="coerce").median())
                iv_atm_new = float(df_new.loc[(df_new["days_to_exp"].between(20, 60)) & (df_new["mny"].abs() <= 0.01), "implied_volatility"].median())
                T_new = max(dte_new, 1) / 365.0
                sigma_pts_new = S_now * iv_atm_new * math.sqrt(T_new)
                smile_new = (df_new.groupby(["type", "strike"], as_index=False)["implied_volatility"].median()
                                .set_index(["type", "strike"])["implied_volatility"].to_dict())

                def get_iv_new(side: str, K: float) -> float:
                    v = smile_new.get((side, K), np.nan)
                    return float(v) if not np.isnan(v) else float(iv_atm_new)

                def nearest_strike_new(side: str, target_price: float) -> float:
                    arr = sorted(df_new[df_new["type"] == side]["strike"].unique().tolist())
                    if not arr:
                        return np.nan
                    return float(min(arr, key=lambda x: abs(float(x) - float(target_price))))

                if roll_mode.startswith("σ"):
                    new_put = nearest_strike_new("put", S_now - 1.2 * sigma_pts_new)
                    new_call = nearest_strike_new("call", S_now + 1.2 * sigma_pts_new)
                else:
                    puts = sorted(df_new[df_new["type"] == "put"]["strike"].unique().tolist())
                    calls = sorted(df_new[df_new["type"] == "call"]["strike"].unique().tolist())
                    bp, bc, ep, ec = np.nan, np.nan, 1e9, 1e9
                    for K in puts:
                        d = bs_delta_vectorized([S_now], [K], [get_iv_new("put", K)], [dte_new], r_input, q_input, [False])[0]
                        e = abs(abs(d) - delta_target)
                        if not np.isnan(d) and e < ep:
                            bp, ep = K, e
                    for K in calls:
                        d = bs_delta_vectorized([S_now], [K], [get_iv_new("call", K)], [dte_new], r_input, q_input, [True])[0]
                        e = abs(d - delta_target)
                        if not np.isnan(d) and e < ec:
                            bc, ec = K, e
                    new_put, new_call = bp, bc

                # nieuwe credit vs sluitkosten
                def _val(df_leg, typ, K):
                    row = df_leg[(df_leg["type"] == typ) & (df_leg["strike"] == K)]
                    return float(pd.to_numeric(row[price_source], errors="coerce").median()) if not row.empty else np.nan

                new_put_px, new_call_px = _val(df_new, "put", new_put), _val(df_new, "call", new_call)
                new_credit = (new_put_px + new_call_px) if (not np.isnan(new_put_px) and not np.isnan(new_call_px)) else np.nan
                close_cost = (float(put_px) if not np.isnan(put_px) else 0.0) + (float(call_px) if not np.isnan(call_px) else 0.0)
                net_roll_credit = (new_credit - close_cost) if (not np.isnan(new_credit)) else np.nan

                def sigma_dist(K, sp):
                    return abs(K - S_now) / sp if (sp and sp > 0 and not np.isnan(sp)) else np.nan

                old_sd_put, old_sd_call = sigma_dist(float(target_put), sigma_pts_new), sigma_dist(float(target_call), sigma_pts_new)
                new_sd_put, new_sd_call = sigma_dist(float(new_put), sigma_pts_new), sigma_dist(float(new_call), sigma_pts_new)

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Nieuwe exp.", str(new_exp))
                r2.metric("Nieuwe strikes", f"P {new_put:.0f} / C {new_call:.0f}")
                r3.metric("Extra credit (roll)", f"{net_roll_credit:,.2f}" if not np.isnan(net_roll_credit) else "—")
                r4.metric("DTE (nieuw)", f"{dte_new:.0f}")
                r5, r6 = st.columns(2)
                r5.metric("σ PUT (oud → nieuw)", f"{old_sd_put:.2f}σ → {new_sd_put:.2f}σ" if not (np.isnan(old_sd_put) or np.isnan(new_sd_put)) else "—")
                r6.metric("σ CALL (oud → nieuw)", f"{old_sd_call:.2f}σ → {new_sd_call:.2f}σ" if not (np.isnan(old_sd_call) or np.isnan(new_sd_call)) else "—")

# ─────────────────────────── Footer ───────────────────────────
st.caption("🔍 Navigatie: zoom met scroll/pinch, **double-click** om te rescalen. OI/Volume-filters helpen spikes te filteren.")
