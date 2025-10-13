# pages/3_SPX_Options.py
# SPX Options Dashboard — skew (points-to-strike), PPD, term structure, strangle helper, margin/payoff, roll-sim

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
from google.oauth2 import service_account
from scipy.stats import norm

# ─────────────────────────── Page / Plot config ───────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧰 SPX Options — Skew, Delta & PPD")

PLOTLY_CONFIG = {
    "scrollZoom": True, "doubleClick": "reset", "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

# ─────────────────────────── BigQuery helpers ─────────────────────────────
def get_bq_client():
    sa = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa)
    return bigquery.Client(project=sa["project_id"], credentials=creds)

_bq = get_bq_client()
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
VIEW = f"{PROJECT_ID}.marketdata.spx_options_enriched_v"  # pas desgewenst aan via secrets

def _bq_param(name, value):
    # kleine helper die types netjes mapt
    if isinstance(value, bool):   return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, int):    return bigquery.ScalarQueryParameter(name, "INT64", value)
    if isinstance(value, float):  return bigquery.ScalarQueryParameter(name, "FLOAT64", value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return bigquery.ScalarQueryParameter(name, "TIMESTAMP", pd.to_datetime(value).to_pydatetime())
    if isinstance(value, (date, )):
        return bigquery.ScalarQueryParameter(name, "DATE", str(value))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=[_bq_param(k, v) for k, v in params.items()])
    return _bq.query(sql, job_config=job_config).to_dataframe()

# ─────────────────────────── Utils ────────────────────────────────────────
def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < 3: return y
    return y.rolling(window, center=True, min_periods=1).median()

def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2: return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))

def ppd_series(df_like: pd.DataFrame, unit: str = "points") -> pd.Series:
    s = pd.to_numeric(df_like["ppd"], errors="coerce")
    if unit.startswith("bp"):
        u = pd.to_numeric(df_like["underlying_price"], errors="coerce")
        s = 10000.0 * s / u
    return s.replace(0.0, np.nan)

def bs_delta_vectorized(S, K, IV, T_days, r, q, is_call):
    """Vectorized Black–Scholes delta (continue r,q), T in dagen."""
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    IV = np.asarray(IV, dtype=float); T = np.asarray(T_days, dtype=float)/365.0
    is_call = np.asarray(is_call, dtype=bool)
    n = len(S)
    r_arr = np.full(n, r, dtype=float); q_arr = np.full(n, q, dtype=float)
    eps = 1e-12
    sigma = np.maximum(IV, eps); sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(np.maximum(S, eps)/np.maximum(K, eps)) + (r_arr - q_arr + 0.5*sigma**2)*T) / (sigma*sqrtT)
    disc = np.exp(-q_arr*T)
    return np.where(is_call, disc*norm.cdf(d1), -disc*norm.cdf(-d1)).astype(float)

def strangle_payoff_at_expiry(S, Kp, Kc, credit_pts, multiplier=100) -> float:
    return (credit_pts - max(Kp - S, 0.0) - max(S - Kc, 0.0)) * multiplier

def span_like_margin(S, Kp, Kc, credit_pts, down=0.15, up=0.10, multiplier=100) -> float:
    S_down, S_up = S*(1-down), S*(1+up)
    loss_down = (max(Kp - S_down, 0.0) - credit_pts) * multiplier
    loss_up   = (max(S_up - Kc, 0.0) - credit_pts) * multiplier
    return float(max(0.0, loss_down, loss_up))

def regt_strangle_margin(S, Kp, Kc, put_px_pts, call_px_pts, multiplier=100) -> float:
    otm_call, otm_put = max(Kc - S, 0.0), max(S - Kp, 0.0)
    base_call = max(0.20*S - otm_call, 0.10*S)
    base_put  = max(0.20*S - otm_put,  0.10*S)
    req_call = (call_px_pts + base_call) * multiplier
    req_put  = (put_px_pts  + base_put ) * multiplier
    worst_leg = max(req_call, req_put)
    other_leg = put_px_pts if worst_leg == req_call else call_px_pts
    return float(worst_leg + other_leg * multiplier)

# ─────────────────────────── Sidebar: Skew settings ──────────────────────
with st.sidebar:
    st.header("⚙️ Instellingen")
    center_mode = st.radio("Strike-centrering", ["Rounded (aanbevolen)", "ATM (live underlying)"], index=0)
    round_base = st.select_slider("Rond strikes op", options=[25, 50, 100], value=25)
    max_pts    = st.slider("Afstand tot (gecentreerde) strike (± punten)", 50, 1000, 400, 50)
    dte_pref   = st.selectbox("DTE-selectie voor skew", ["Nearest", "0–7", "8–21", "22–45", "46–90", "90+"])
    r_input    = st.number_input("Risicovrije rente r (p.j.)", value=0.00, step=0.25)
    q_input    = st.number_input("Dividend/Index carry q (p.j.)", value=0.00, step=0.25)
    st.caption("Tip: **Rounded + ±400–600** geeft stabiele skew-curves. ATM kan intraday verschuiven.")

with st.expander("ℹ️ Uitleg: points-to-strike, ronding en DTE"):
    st.markdown(
        "- **Points to strike** = `K − S₀` in punten. Negatief = put-zijde, positief = call-zijde.\n"
        "- **Rounded** centreert rond een afgeronde S₀ (25/50/100) → rustigere as.\n"
        "- **ATM** centreert exact op de actuele S.\n"
        "- **DTE**: kies één bucket voor skew; de rest zie je in Term Structure."
    )

# ─────────────────────────── Data: laatste snapshot ──────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_latest_snapshot() -> pd.DataFrame:
    sql = f"""
    WITH last AS (SELECT MAX(snapshot_date) AS snapshot_date FROM `{VIEW}`)
    SELECT contract_symbol, type, expiration, days_to_exp, strike, underlying_price,
           last_price, bid, ask, mid_price, implied_volatility, open_interest, volume,
           vix, snapshot_date
    FROM `{VIEW}`
    WHERE snapshot_date = (SELECT snapshot_date FROM last)
    """
    return run_query(sql)

df = load_latest_snapshot()
if df.empty:
    st.warning("Geen SPX-optiedata gevonden in de view.")
    st.stop()

# Clean
num_cols = ["strike","underlying_price","implied_volatility","days_to_exp","open_interest","volume","last_price","bid","ask","mid_price"]
for c in num_cols:
    if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["strike","underlying_price","implied_volatility","days_to_exp"]).copy()
df["type"] = df["type"].str.lower()

# Centreren in punten
S_now = float(np.nanmedian(df["underlying_price"]))
center = round_base * round(S_now/round_base) if center_mode.startswith("Rounded") else S_now
df["pts_to_strike"] = df["strike"] - center
df = df[df["pts_to_strike"].between(-max_pts, max_pts)].copy()
if df.empty:
    st.warning("Geen rijen binnen de ingestelde ± afstand.")
    st.stop()

# Keuze DTE-bucket voor skew
if dte_pref == "Nearest":
    target_dte = float(df["days_to_exp"].min())
    skew_df = df.loc[df["days_to_exp"] == target_dte].copy()
else:
    lo, hi = {"0–7":(0,7), "8–21":(8,21), "22–45":(22,45), "46–90":(46,90), "90+":(90, 10_000)}[dte_pref]
    skew_df = df.loc[df["days_to_exp"].between(lo, hi)].copy()
if skew_df.empty:
    st.warning("Geen rijen in de gekozen DTE-bucket.")
    st.stop()

# Delta vectorized
skew_df["is_call"] = skew_df["type"].eq("call")
skew_df["delta"] = bs_delta_vectorized(
    skew_df["underlying_price"], skew_df["strike"], skew_df["implied_volatility"],
    skew_df["days_to_exp"], r_input, q_input, skew_df["is_call"]
)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Underlying (S₀)", f"{S_now:,.0f}")
c2.metric("Center", f"{center:,.0f}")
c3.metric("DTE (skew)", f"{skew_df['days_to_exp'].median():.0f} d")
c4.metric("Rijen (skew)", f"{len(skew_df):,}")

# ─────────────────────────── Skew plots ─────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Skew (IV & Δ vs punten)",
    "🧵 Term Structure",
    "🧮 PPD per serie / afstand / DTE",
    "🗺️ Matrix (heatmap)",
    "📊 Vol & Risk",
    "🔧 Strangle Helper + Margin/Payoff + Roll"
])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Implied Volatility (IV) vs Points to strike",
                                        "Delta (Δ) vs Points to strike"))
    for side in ["call","put"]:
        sub = skew_df[skew_df["type"]==side]
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub["pts_to_strike"], y=sub["implied_volatility"], mode="markers",
                                 name=f"IV {side}", hovertemplate="pts: %{x:.0f}<br>IV: %{y:.1%}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub["pts_to_strike"], y=sub["delta"], mode="markers",
                                 name=f"Δ {side}", hovertemplate="pts: %{x:.0f}<br>Δ: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Points to strike (K − center)", row=2, col=1)
    fig.update_yaxes(title_text="IV", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Δ", row=2, col=1)
    fig.update_layout(height=680, showlegend=True, margin=dict(t=60,b=40,l=40,r=20))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────── Term Structure & Smile ────────────────────
with tab2:
    # Term structure per points-band
    bins = [-10_000, -200, -50, 50, 200, 10_000]
    labels = ["Put far (≤−200)","Put near (−200..−50)","Near ATM (−50..50)","Call near (50..200)","Call far (≥200)"]
    df["pts_band"] = pd.cut(df["pts_to_strike"], bins=bins, labels=labels)
    ts = df.groupby(["days_to_exp","pts_band"], as_index=False)["implied_volatility"].median().dropna()
    fig_ts = go.Figure()
    for band in labels:
        sub = ts.loc[ts["pts_band"] == band]
        if sub.empty: continue
        fig_ts.add_trace(go.Scatter(x=sub["days_to_exp"], y=sub["implied_volatility"], mode="lines+markers", name=band,
                                    hovertemplate="DTE: %{x:.0f}d<br>IV: %{y:.1%}<extra></extra>"))
    fig_ts.update_layout(height=420, xaxis_title="DTE (dagen)", yaxis_title="Median IV", yaxis_tickformat=".0%")
    st.plotly_chart(fig_ts, use_container_width=True)

    # IV Smile op laatste snapshot
    exps = sorted(pd.to_datetime(df["expiration"]).dt.date.unique().tolist())
    exp_for_smile = st.selectbox("Expiratie voor IV Smile", options=exps or [None], index=0)
    df_last = df.copy()  # (we zitten al op laatste snapshot)
    sm = df_last[df_last["expiration"].astype(str) == str(exp_for_smile)]
    if sm.empty:
        st.info("Geen data voor gekozen expiratie.")
    else:
        sm = sm.groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
        if len(sm) >= 5:
            lo, hi = sm["implied_volatility"].quantile([0.02, 0.98])
            sm["implied_volatility"] = sm["implied_volatility"].clip(lower=lo, upper=hi)
        fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV"))
        fig_sm.update_layout(height=420, xaxis_title="Strike", yaxis_title="Implied Volatility", yaxis_tickformat=".0%")
        st.plotly_chart(fig_sm, use_container_width=True)

# ─────────────────────────── PPD & analyses ────────────────────────────
with tab3:
    st.subheader("Serie door de tijd")
    # Periodekeuze over historie
    @st.cache_data(ttl=600, show_spinner=False)
    def load_bounds():
        q = f"SELECT MIN(CAST(snapshot_date AS DATE)) min_d, MAX(CAST(snapshot_date AS DATE)) max_d FROM `{VIEW}`"
        b = run_query(q)
        return pd.to_datetime(b["min_d"].iloc[0]).date(), pd.to_datetime(b["max_d"].iloc[0]).date()
    min_d, max_d = load_bounds()
    start_date, end_date = st.date_input("Periode (snapshot_date)", value=(max(min_d, max_d - timedelta(days=365)), max_d),
                                         min_value=min_d, max_value=max_d, format="YYYY-MM-DD")
    sel_type = st.radio("Type", ["call","put"], index=1, horizontal=True)
    dte_min, dte_max = st.slider("DTE", 0, 365, (0, 60), step=1)
    mny_min, mny_max = st.slider("Moneyness (K/S − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)

    @st.cache_data(ttl=600, show_spinner=True)
    def load_filtered(start_date, end_date, sel_type, dte_min, dte_max, mny_min, mny_max):
        sql = f"""
        WITH base AS (
          SELECT snapshot_date, contract_symbol, type, expiration, days_to_exp, strike, underlying_price,
                 SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price,0)) - 1.0 AS moneyness,
                 (CAST(strike AS FLOAT64) - CAST(underlying_price AS FLOAT64)) AS dist_points,
                 last_price, bid, ask, mid_price, implied_volatility, open_interest, volume, vix, ppd
          FROM `{VIEW}`
          WHERE DATE(snapshot_date) BETWEEN @start AND @end
            AND LOWER(type) = @t
            AND days_to_exp BETWEEN @dte_min AND @dte_max
            AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price,0)) - 1.0 BETWEEN @mny_min AND @mny_max
        )
        SELECT * FROM base
        """
        params = {"start": start_date, "end": end_date, "t": sel_type,
                  "dte_min": int(dte_min), "dte_max": int(dte_max),
                  "mny_min": float(mny_min), "mny_max": float(mny_max)}
        df = run_query(sql, params=params)
        if df.empty: return df
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        for c in ["days_to_exp","implied_volatility","open_interest","volume","ppd","strike",
                  "underlying_price","last_price","mid_price","bid","ask","dist_points","moneyness"]:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["moneyness_pct"] = 100 * df["moneyness"]
        df["abs_dist_pct"] = (np.abs(df["dist_points"]) / df["underlying_price"]) * 100.0
        df["snap_min"] = df["snapshot_date"].dt.floor("min")
        return df

    dfh = load_filtered(start_date, end_date, sel_type, dte_min, dte_max, mny_min, mny_max)
    if dfh.empty:
        st.warning("Geen data in deze filters.")
        st.stop()

    # Liquiditeitfilters
    cL1, cL2, cL3 = st.columns(3)
    with cL1: min_oi = st.slider("Min Open Interest", 0, 50, 1, step=1)
    with cL2: min_vol = st.slider("Min Volume", 0, 50, 1, step=1)
    with cL3: min_per_bin = st.slider("Min punten per bin (aggr)", 1, 10, 3, step=1)
    liq_mask = ((dfh["open_interest"].fillna(0) >= min_oi) | (dfh["volume"].fillna(0) >= min_vol))

    # Defaults
    snaps = sorted(dfh["snap_min"].unique())
    default_snapshot = snaps[-1] if snaps else None
    underlying_now = float(dfh.loc[dfh["snap_min"]==default_snapshot, "underlying_price"].mean()) if default_snapshot else float(df["underlying_price"].median())

    # Serie-selectie
    st.markdown("#### Serie-selectie — volg één optiereeks door de tijd")
    strikes_all = sorted([float(x) for x in dfh["strike"].dropna().unique().tolist()])
    exps_all = sorted(pd.to_datetime(dfh["expiration"]).dt.date.unique().tolist())
    def pick_closest(options, target):
        return min(options, key=lambda x: abs(float(x)-float(target))) if options else None
    default_series_exp = pick_closest(exps_all, date.today() + timedelta(days=14)) or (exps_all[0] if exps_all else None)
    default_series_strike = pick_closest(strikes_all, (underlying_now - 300 if sel_type=="put" else underlying_now + 200)) or (strikes_all[0] if strikes_all else 6000)

    cS1, cS2, cS3 = st.columns([1,1,1.4])
    with cS1:
        series_strike = st.selectbox("Serie Strike", options=strikes_all or [6000.0],
                                     index=(strikes_all.index(default_series_strike) if strikes_all and default_series_strike in strikes_all else 0))
    with cS2:
        series_exp = st.selectbox("Serie Expiratie", options=exps_all or [date.today()],
                                  index=(exps_all.index(default_series_exp) if exps_all and default_series_exp in exps_all else 0))
    with cS3:
        price_col = st.radio("Prijsbron", ["last_price","mid_price"], index=0, horizontal=True)

    serie = dfh[(dfh["strike"]==series_strike) & (dfh["expiration"]==series_exp) & liq_mask].copy().sort_values("snapshot_date")
    if serie.empty:
        st.info("Geen (genoeg) liquiditeit voor deze combinatie binnen de huidige filters.")
    else:
        a1, a2 = st.columns(2)
        with a1:
            fig_price = make_subplots(specs=[[{"secondary_y": True}]])
            fig_price.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie[price_col], name="Price", mode="lines+markers", connectgaps=True), secondary_y=False)
            fig_price.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"], name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True), secondary_y=True)
            fig_price.update_layout(title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | Price vs SP500", height=420, hovermode="x unified")
            fig_price.update_xaxes(title_text="Meetmoment")
            fig_price.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
            fig_price.update_yaxes(title_text="SP500", secondary_y=True)
            st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_CONFIG)
        with a2:
            unit = st.radio("PPD-eenheid", ["Points per day", "bp/day (vs onderliggende)"], index=0, horizontal=True)
            fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"], y=ppd_series(serie, unit="bp" if unit.startswith("bp") else "points"),
                                         name="PPD", mode="lines+markers", connectgaps=True), secondary_y=False)
            fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"], name="SP500", mode="lines",
                                         line=dict(dash="dot"), connectgaps=True), secondary_y=True)
            fig_ppd.update_layout(title=f"{sel_type.UPPER()} {series_strike} — exp {series_exp} | PPD vs SP500", height=420, hovermode="x unified")
            fig_ppd.update_xaxes(title_text="Meetmoment")
            fig_ppd.update_yaxes(title_text=("PPD (bp/day)" if unit.startswith("bp") else "PPD (points/day)"), secondary_y=False, rangemode="tozero")
            fig_ppd.update_yaxes(title_text="SP500", secondary_y=True)
            st.plotly_chart(fig_ppd, use_container_width=True, config=PLOTLY_CONFIG)

    # PPD vs Afstand (laatste snapshot)
    st.markdown("#### PPD vs Afstand (laatste snapshot)")
    sel_snapshot = st.selectbox("Peildatum (snapshot)", options=snaps, index=(len(snaps)-1 if snaps else 0),
                                format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"))
    df_last = dfh[(dfh["snap_min"] == sel_snapshot) & liq_mask].copy()
    if df_last.empty:
        st.info("Geen data op dit snapshot (na liquiditeit-filter).")
    else:
        unit2 = st.radio("PPD-eenheid (grafiek)", ["Points per day", "bp/day"], index=0, horizontal=True, key="ppd_unit2")
        df_last["ppd_u"] = ppd_series(df_last, unit="bp" if unit2.startswith("bp") else "points")
        df_last["abs_dist_pct"] = ((df_last["dist_points"].abs() / df_last["underlying_price"]) * 100.0)
        bins = np.arange(0, 18.5, 0.5)
        df_last["dist_bin"] = pd.cut(df_last["abs_dist_pct"], bins=bins, include_lowest=True)
        g = (df_last.groupby("dist_bin").agg(ppd=("ppd_u","median"), n=("ppd_u","count")).reset_index())
        g = g[g["n"] >= min_per_bin].copy()
        g["bin_mid"] = g["dist_bin"].apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
        g = g.dropna(subset=["bin_mid"]).sort_values("bin_mid")
        g["ppd_s"] = smooth_series(g["ppd"], window=3)
        best_idx = g["ppd_s"].idxmax() if not g.empty else None

        fig_ppd_dist = go.Figure()
        fig_ppd_dist.add_vrect(x0=-0.5, x1=0.5, fillcolor="lightgrey", opacity=0.25, line_width=0,
                               annotation_text="ATM-zone", annotation_position="top left")
        fig_ppd_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd"], mode="markers", name="PPD (median/bin)", opacity=0.85))
        fig_ppd_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd_s"], mode="lines", name="Smoothed"))
        if best_idx is not None and pd.notna(g.loc[best_idx,"ppd_s"]):
            x_b, y_b = float(g.loc[best_idx,"bin_mid"]), float(g.loc[best_idx,"ppd_s"])
            fig_ppd_dist.add_annotation(x=x_b, y=y_b, text=f"sweet spot ≈ {y_b:.2f} @ {x_b:.2f}%", showarrow=True, arrowhead=2)
        fig_ppd_dist.update_layout(title=f"PPD vs Afstand — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
                                   xaxis_title="Afstand |K−S|/S (%)",
                                   yaxis_title=("PPD (bp/day)" if unit2.startswith("bp") else "PPD (points/day)"),
                                   height=420)
        st.plotly_chart(fig_ppd_dist, use_container_width=True, config=PLOTLY_CONFIG)

    # PPD vs DTE
    st.markdown("#### PPD vs DTE — opbouw premium/dag")
    cD1, cD2, cD3, cD4 = st.columns(4)
    with cD1: mode = st.radio("Bereik", ["ATM-band (moneyness)", "Rond gekozen strike"], index=0)
    with cD2: atm_band = st.slider("ATM-band (± moneyness %)", 0.01, 0.10, 0.02, step=0.01)
    with cD3: strike_window = st.slider("Strike-venster (punten)", 10, 200, 50, step=10)
    with cD4: robust = st.checkbox("Robust scale (95e pct)", value=True)
    base = df_last if True else dfh
    if mode.startswith("ATM"):
        df_ppd = base[np.abs(base["moneyness"]) <= atm_band].copy()
    else:
        df_ppd = base[(base["strike"] >= series_strike - strike_window) & (base["strike"] <= series_strike + strike_window)].copy()
    if df_ppd.empty:
        st.info("Geen data voor PPD vs DTE met deze instellingen.")
    else:
        unit3 = st.radio("PPD-eenheid (DTE)", ["Points per day", "bp/day"], index=0, horizontal=True, key="ppd_unit3")
        df_ppd = df_ppd.assign(ppd_u=ppd_series(df_ppd, unit="bp" if unit3.startswith("bp") else "points"))
        curve = (df_ppd.groupby("days_to_exp", as_index=False).agg(ppd=("ppd_u","median"), n=("ppd_u","count"))
                 .query("n >= @min_per_bin").sort_values("days_to_exp"))
        curve["ppd_s"] = smooth_series(curve["ppd"], window=3)
        y_range = None
        if robust and curve["ppd"].notna().any():
            hi = float(np.nanpercentile(curve["ppd"], 95)); lo = float(np.nanpercentile(curve["ppd"], 5)); pad = (hi-lo)*0.10
            y_range = [max(lo-pad, 0.0), hi+pad]
        fig_ppd_dte = go.Figure()
        fig_ppd_dte.add_trace(go.Scatter(x=curve["days_to_exp"], y=curve["ppd"], mode="markers", name="PPD (median)", opacity=0.85))
        fig_ppd_dte.add_trace(go.Scatter(x=curve["days_to_exp"], y=curve["ppd_s"], mode="lines", name="Smoothed"))
        fig_ppd_dte.update_layout(title="PPD vs Days To Expiration", xaxis_title="DTE", yaxis_title=("PPD (bp/day)" if unit3.startswith("bp") else "PPD (points/day)"), height=420)
        if y_range: fig_ppd_dte.update_yaxes(range=y_range)
        st.plotly_chart(fig_ppd_dte, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Matrix ─────────────────────────────────────
with tab4:
    st.subheader("Matrix — meetmoment × strike (laatste snapshot per minuut)")
    # kies expiratie
    exps_all = sorted(pd.to_datetime(df["expiration"]).dt.date.unique().tolist())
    matrix_exp = st.selectbox("Expiratie (matrix)", options=exps_all or [None], index=0)
    matrix_metric = st.radio("Waarde", ["last_price","mid_price","ppd"], horizontal=True, index=0)
    max_rows = st.slider("Max. meetmomenten (recentste)", 50, 500, 200, step=50)

    # slice voor exp (laatste snapshot al geladen: df)
    mx = df[df["expiration"].astype(str) == str(matrix_exp)].copy()
    if mx.empty:
        st.info("Geen matrix-data voor de gekozen expiratie.")
    else:
        if matrix_metric == "ppd":
            # ppd is al in view aanwezig; zo niet, kun je mid/dte gebruiken
            pass
        mx["snap_s"] = pd.to_datetime(mx["snapshot_date"]).dt.strftime("%Y-%m-%d %H:%M")
        pv = (mx.groupby(["snap_s","strike"], as_index=False)[matrix_metric].median()
                .pivot(index="snap_s", columns="strike", values=matrix_metric)
                .sort_index(ascending=False).head(max_rows)).round(3)
        arr = pv.values.astype(float)
        tab_hm, tab_tbl = st.tabs(["Heatmap", "Tabel"])
        with tab_hm:
            fig_mx = go.Figure(data=go.Heatmap(z=arr, x=pv.columns.astype(float), y=pv.index.tolist(), colorbar_title=matrix_metric))
            fig_mx.update_layout(height=520, xaxis_title="Strike", yaxis_title="Meetmoment")
            st.plotly_chart(fig_mx, use_container_width=True, config=PLOTLY_CONFIG)
        with tab_tbl:
            st.dataframe(pv, use_container_width=True)

# ─────────────────────────── Vol & Risk ────────────────────────────────
with tab5:
    st.subheader("📊 Vol & Risk (ATM-IV, HV, VRP, IV-Rank, Expected Move)")
    # Daily underlying serie
    u_daily = (df.assign(dte=df["snapshot_date"])
                 .sort_values("dte").groupby(df["snapshot_date"].dt.date, as_index=False)
                 .agg(close=("underlying_price","median")))
    u_daily["ret"] = u_daily["close"].pct_change()
    hv20 = annualize_std(u_daily["ret"].tail(21).dropna())

    near_atm = df[(df["days_to_exp"].between(20, 40)) & (df["strike"]/df["underlying_price"] - 1.0).abs().le(0.01)]
    iv_atm = float(near_atm["implied_volatility"].median()) if not near_atm.empty else float(df["implied_volatility"].median())

    iv_hist = (df[(df["days_to_exp"].between(20,40)) & ((df["strike"]/df["underlying_price"] - 1.0).abs().le(0.01))]
               .groupby(df["snapshot_date"].dt.date, as_index=False)["implied_volatility"].median()
               .rename(columns={"implied_volatility":"iv"}))
    iv_1y = iv_hist["iv"].tail(252) if not iv_hist.empty else pd.Series(dtype=float)
    iv_rank = float((iv_1y <= iv_1y.iloc[-1]).mean()) if len(iv_1y) >= 2 else np.nan

    dte_selected = int(pd.to_numeric(df["days_to_exp"], errors="coerce").median()) if not df.empty else 30
    em_sigma = (S_now * iv_atm * math.sqrt(max(dte_selected,1)/365.0)) if (not np.isnan(S_now) and not np.isnan(iv_atm)) else np.nan

    cV1, cV2, cV3, cV4, cV5 = st.columns(5)
    cV1.metric("ATM-IV (~30D)", f"{iv_atm:.2%}" if not np.isnan(iv_atm) else "—")
    cV2.metric("HV20", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
    cV3.metric("VRP (IV−HV)", f"{(iv_atm-hv20):.2%}" if (not np.isnan(iv_atm) and not np.isnan(hv20)) else "—")
    cV4.metric("IV-Rank (1y)", f"{iv_rank*100:.0f}%" if not np.isnan(iv_rank) else "—")
    cV5.metric("Expected Move (σ)", f"±{em_sigma:,.0f} pts ({em_sigma/S_now:.2%})" if (not np.isnan(em_sigma) and S_now>0) else "—")

    # Put/Call-ratio
    st.subheader("Put/Call-ratio per expiratie")
    p = (df.groupby(["expiration","type"], as_index=False)
           .agg(vol=("volume","sum"), oi=("open_interest","sum")))
    if p.empty:
        st.info("Geen data voor PCR.")
    else:
        pv = (p.pivot_table(index="expiration", columns="type", values=["vol","oi"], aggfunc="sum")
                .sort_index().sort_index(axis=1)).fillna(0.0)
        pv.columns = [f"{a}_{b}" for a,b in pv.columns.to_flat_index()]
        for col in ["vol_put","vol_call","oi_put","oi_call"]:
            if col not in pv.columns: pv[col] = 0.0
        pv["PCR_vol"] = pv["vol_put"] / pv["vol_call"].replace(0, np.nan)
        pv["PCR_oi"]  = pv["oi_put"]  / pv["oi_call"].replace(0, np.nan)
        pv = pv.replace([np.inf,-np.inf], np.nan).dropna(subset=["PCR_vol","PCR_oi"], how="all")
        if pv.empty:
            st.info("Niet genoeg data (alleen puts of alleen calls in de selectie).")
        else:
            fig_pcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    subplot_titles=("PCR op Volume", "PCR op Open Interest"))
            fig_pcr.add_trace(go.Bar(x=pv.index, y=pv["PCR_vol"], name="PCR (Vol)"), row=1, col=1)
            fig_pcr.add_trace(go.Bar(x=pv.index, y=pv["PCR_oi"],  name="PCR (OI)"),  row=2, col=1)
            fig_pcr.update_layout(height=520, title_text="Put/Call-ratio per Expiratie")
            st.plotly_chart(fig_pcr, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Strangle Helper / Margin / Roll ───────────
with tab6:
    st.subheader("🧠 Strangle Helper (σ- of Δ-doel) + 💳 Margin & Payoff + 🔄 Roll")
    # Slice kiezen: bestaande df = laatste snapshot; gebruik expiration keuze + strikes
    exps_all = sorted(pd.to_datetime(df["expiration"]).dt.date.unique().tolist())
    if not exps_all:
        st.info("Geen expiraties beschikbaar op dit snapshot.")
        st.stop()

    cm1, cm2, cm3, cm4 = st.columns([1.2, 1, 1, 1])
    with cm1: str_sel_mode = st.radio("Selectiemodus", ["σ-doel", "Δ-doel"], index=0)
    with cm2: sigma_target = st.slider("σ-doel per zijde", 0.5, 2.5, 1.0, step=0.1)
    with cm3: delta_target = st.slider("Δ-doel (absoluut)", 0.05, 0.30, 0.15, step=0.01)
    with cm4: price_source = st.radio("Prijsbron", ["mid_price","last_price"], index=0, horizontal=True)

    cexp1, cexp2 = st.columns([1.2,1])
    with cexp1:
        exp_for_str = st.selectbox("Expiratie voor strangle", options=exps_all, index=min(0, len(exps_all)-1))
    with cexp2:
        use_smile_iv = st.checkbox("Gebruik strike-IV (smile) voor Δ", value=False)

    df_str = df[df["expiration"].astype(str) == str(exp_for_str)].copy()
    df_str["mny"] = df_str["strike"]/df_str["underlying_price"] - 1.0
    df_str = df_str[((df_str["open_interest"].fillna(0)>=1) | (df_str["volume"].fillna(0)>=1))]
    if df_str.empty:
        st.info("Geen liquide rijen voor deze expiratie.")
        st.stop()

    iv_atm_exp = float(df_str.loc[(df_str["days_to_exp"].between(20,60)) & (df_str["mny"].abs()<=0.01),"implied_volatility"].median())
    dte_exp = int(pd.to_numeric(df_str["days_to_exp"], errors="coerce").median())
    T = max(dte_exp,1)/365.0
    sigma_pts = S_now * iv_atm_exp * math.sqrt(T)
    smile_map = (df_str.groupby(["type","strike"], as_index=False)["implied_volatility"].median()
                 .set_index(["type","strike"])["implied_volatility"].to_dict())

    def get_iv(side: str, K: float) -> float:
        if use_smile_iv:
            v = smile_map.get((side, K), np.nan)
            if not np.isnan(v): return float(v)
        return float(iv_atm_exp)

    def nearest_strike(side: str, target_price: float) -> float:
        s_list = sorted(df_str[df_str["type"]==side]["strike"].unique().tolist())
        if not s_list: return np.nan
        return float(min(s_list, key=lambda x: abs(float(x)-float(target_price))))

    def pick_by_sigma():
        if np.isnan(sigma_pts): return np.nan, np.nan
        return (nearest_strike("put",  S_now - sigma_target*sigma_pts),
                nearest_strike("call", S_now + sigma_target*sigma_pts))

    def pick_by_delta():
        puts  = sorted(df_str[df_str["type"]=="put"]["strike"].unique().tolist())
        calls = sorted(df_str[df_str["type"]=="call"]["strike"].unique().tolist())
        best_p, best_c, err_p, err_c = np.nan, np.nan, 1e9, 1e9
        for K in puts:
            d = bs_delta_vectorized([S_now],[K],[get_iv("put",K)],[dte_exp], r_input, q_input, [False])[0]
            e = abs(abs(d) - delta_target)
            if not np.isnan(d) and e < err_p: best_p, err_p = K, e
        for K in calls:
            d = bs_delta_vectorized([S_now],[K],[get_iv("call",K)],[dte_exp], r_input, q_input, [True])[0]
            e = abs(d - delta_target)
            if not np.isnan(d) and e < err_c: best_c, err_c = K, e
        return float(best_p), float(best_c)

    target_put, target_call = (pick_by_sigma() if str_sel_mode.startswith("σ") else pick_by_delta())

    def _px(typ, K):
        row = df_str[(df_str["type"]==typ) & (df_str["strike"]==K)]
        return float(pd.to_numeric(row[price_source], errors="coerce").median()) if not row.empty else np.nan

    put_px, call_px = _px("put", target_put), _px("call", target_call)
    total_credit = (put_px + call_px) if (not np.isnan(put_px) and not np.isnan(call_px)) else np.nan

    def sigma_distance(K: float) -> float:
        return abs(K - S_now) / sigma_pts if (sigma_pts and not np.isnan(sigma_pts)) else np.nan

    sd_put, sd_call = sigma_distance(target_put), sigma_distance(target_call)

    # touch-proxy (ruime benadering)
    def p_itm_at_exp(sd: float) -> float:
        return (1.0 - norm.cdf(sd)) if not np.isnan(sd) else np.nan
    p_touch_put  = min(1.0, 2.0 * p_itm_at_exp(sd_put))  if not np.isnan(sd_put)  else np.nan
    p_touch_call = min(1.0, 2.0 * p_itm_at_exp(sd_call)) if not np.isnan(sd_call) else np.nan
    p_both_touch = min(1.0, (p_touch_put or 0.0) + (p_touch_call or 0.0))

    ppd_total_pts = float(total_credit / max(dte_exp,1)) if (not np.isnan(total_credit) and not np.isnan(dte_exp)) else np.nan

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Expiratie", str(exp_for_str))
    k2.metric("DTE", f"{dte_exp:.0f}")
    k3.metric("Strikes", (f"P {target_put:.0f} / C {target_call:.0f}") if not (np.isnan(target_put) or np.isnan(target_call)) else "—")
    k4.metric("Credit (pts)", f"{total_credit:,.2f}" if not np.isnan(total_credit) else "—")
    k5.metric("PPD (pts, 1x)", f"{ppd_total_pts:,.2f}" if not np.isnan(ppd_total_pts) else "—")
    k6.metric("~P(touch) max", f"{p_both_touch*100:.0f}%" if not np.isnan(p_both_touch) else "—")

    # Margin & payoff
    st.markdown("### 💳 Margin & Payoff")
    ready = (not any(np.isnan(x) for x in [S_now, target_put, target_call])) and (total_credit is not np.nan)
    if not ready:
        st.info("Kies eerst een geldige strangle (σ of Δ).")
    else:
        sm1, sm2, sm3, sm4 = st.columns([1.1,1,1,1])
        with sm1: margin_model = st.radio("Margin model", ["SPAN-like stress","Reg-T approx"], index=0)
        with sm2: down_shock = st.slider("Down shock (%)", 5, 30, 15, step=1)
        with sm3: up_shock   = st.slider("Up shock (%)", 5, 30, 10, step=1)
        with sm4: multiplier = st.number_input("Contract multiplier", min_value=10, max_value=250, value=100, step=10)
        if margin_model.startswith("SPAN"):
            est_margin = span_like_margin(S_now, float(target_put), float(target_call), float(total_credit),
                                          down=down_shock/100.0, up=up_shock/100.0, multiplier=multiplier)
        else:
            est_margin = regt_strangle_margin(S_now, float(target_put), float(target_call),
                                              float(put_px) if not np.isnan(put_px) else 0.0,
                                              float(call_px) if not np.isnan(call_px) else 0.0,
                                              multiplier=multiplier)
        n_contracts = int(np.floor(10000.0 / est_margin)) if est_margin>0 else 0  # referentiebudget 10k
        tot_credit_cash = (float(total_credit) if not np.isnan(total_credit) else 0.0) * multiplier
        credit_per_margin = (tot_credit_cash/est_margin) if est_margin>0 else np.nan
        ppd_per_margin = ((ppd_total_pts*multiplier)/est_margin) if (est_margin>0 and not np.isnan(ppd_total_pts)) else np.nan

        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Est. margin (1x)", f"{est_margin:,.0f}")
        mm2.metric("# Contracts @10k", f"{n_contracts:,}")
        mm3.metric("Credit (1x)", f"{tot_credit_cash:,.0f}")
        mm4.metric("Credit / Margin", f"{credit_per_margin:.2f}" if not np.isnan(credit_per_margin) else "—")

        show_payoff = st.checkbox("Toon payoff (1x)", value=True)
        if show_payoff:
            rng = 0.25
            S_grid = np.linspace(S_now*(1-rng), S_now*(1+rng), 400)
            pnl_grid = [strangle_payoff_at_expiry(s, float(target_put), float(target_call), float(total_credit), multiplier=multiplier) for s in S_grid]
            be_low  = float(target_put)  - float(total_credit)
            be_high = float(target_call) + float(total_credit)
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=S_grid, y=pnl_grid, mode="lines", name="PNL @ expiry (1x)"))
            fig_pay.add_hline(y=0, line=dict(dash="dot"))
            fig_pay.add_vline(x=be_low,  line=dict(dash="dot"), annotation_text=f"BE low ≈ {be_low:.0f}")
            fig_pay.add_vline(x=be_high, line=dict(dash="dot"), annotation_text=f"BE high ≈ {be_high:.0f}")
            fig_pay.update_layout(height=420, title=f"Payoff @ Expiry (P {target_put:.0f} / C {target_call:.0f} | credit {total_credit:.2f} pts)",
                                  xaxis_title="S (onderliggende)", yaxis_title=f"PNL (1x, multiplier {multiplier})")
            st.plotly_chart(fig_pay, use_container_width=True, config=PLOTLY_CONFIG)

    # Roll-simulator
    st.markdown("### 🔄 Roll-simulator")
    future_exps = [e for e in exps_all if e > exp_for_str]
    if not future_exps:
        st.info("Geen latere expiraties beschikbaar binnen je filters.")
    elif not ready:
        st.info("Selecteer eerst een strangle in de Strangle Helper.")
    else:
        rr1, rr2 = st.columns([1.2,1])
        with rr1: roll_mode = st.radio("Rol-methode", ["σ-doel","Δ-doel"], index=0, horizontal=True)
        with rr2: new_exp = st.selectbox("Naar welke expiratie rollen?", options=future_exps, index=0)

        df_new = df[df["expiration"].astype(str) == str(new_exp)].copy()
        df_new["mny"] = df_new["strike"]/df_new["underlying_price"] - 1.0
        df_new = df_new[((df_new["open_interest"].fillna(0)>=1) | (df_new["volume"].fillna(0)>=1))]
        if df_new.empty:
            st.info("Geen data voor de gekozen nieuwe expiratie.")
        else:
            dte_new = int(pd.to_numeric(df_new["days_to_exp"], errors="coerce").median())
            iv_atm_new = float(df_new.loc[(df_new["days_to_exp"].between(20,60)) & (df_new["mny"].abs()<=0.01),"implied_volatility"].median())
            T_new = max(dte_new,1)/365.0
            sigma_pts_new = S_now * iv_atm_new * math.sqrt(T_new)
            smile_new = (df_new.groupby(["type","strike"], as_index=False)["implied_volatility"].median()
                            .set_index(["type","strike"])["implied_volatility"].to_dict())
            def get_iv_new(side: str, K: float) -> float:
                return float(smile_new.get((side,K), iv_atm_new))
            def nearest_strike_new(side: str, target_price: float) -> float:
                arr = sorted(df_new[df_new["type"]==side]["strike"].unique().tolist())
                if not arr: return np.nan
                return float(min(arr, key=lambda x: abs(float(x)-float(target_price))))
            if roll_mode.startswith("σ"):
                new_put  = nearest_strike_new("put",  S_now - 1.2 * sigma_pts_new)
                new_call = nearest_strike_new("call", S_now + 1.2 * sigma_pts_new)
            else:
                puts  = sorted(df_new[df_new["type"]=="put"]["strike"].unique().tolist())
                calls = sorted(df_new[df_new["type"]=="call"]["strike"].unique().tolist())
                bp, bc, ep, ec = np.nan, np.nan, 1e9, 1e9
                for K in puts:
                    d = bs_delta_vectorized([S_now],[K],[get_iv_new("put",K)],[dte_new], r_input, q_input, [False])[0]
                    e = abs(abs(d) - delta_target)
                    if not np.isnan(d) and e < ep: bp, ep = K, e
                for K in calls:
                    d = bs_delta_vectorized([S_now],[K],[get_iv_new("call",K)],[dte_new], r_input, q_input, [True])[0]
                    e = abs(d - delta_target)
                    if not np.isnan(d) and e < ec: bc, ec = K, e
                new_put, new_call = bp, bc

            # credits
            def _val(df_leg, typ, K): 
                row = df_leg[(df_leg["type"]==typ) & (df_leg["strike"]==K)]
                return float(pd.to_numeric(row[price_source], errors="coerce").median()) if not row.empty else np.nan
            new_put_px, new_call_px = _val(df_new,"put",new_put), _val(df_new,"call",new_call)
            new_credit = (new_put_px + new_call_px) if (not np.isnan(new_put_px) and not np.isnan(new_call_px)) else np.nan
            close_cost = (float(put_px) if not np.isnan(put_px) else 0.0) + (float(call_px) if not np.isnan(call_px) else 0.0)
            net_roll_credit = (new_credit - close_cost) if (not np.isnan(new_credit)) else np.nan

            def sigma_dist(K, sp): 
                return abs(K - S_now)/sp if (sp and sp>0 and not np.isnan(sp)) else np.nan
            old_sd_put, old_sd_call = sigma_dist(float(target_put), sigma_pts_new), sigma_dist(float(target_call), sigma_pts_new)
            new_sd_put, new_sd_call = sigma_dist(float(new_put),  sigma_pts_new), sigma_dist(float(new_call),  sigma_pts_new)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Nieuwe exp.", str(new_exp))
            r2.metric("Nieuwe strikes", f"P {new_put:.0f} / C {new_call:.0f}")
            r3.metric("Extra credit (roll)", f"{net_roll_credit:,.2f}" if not np.isnan(net_roll_credit) else "—")
            r4.metric("DTE (nieuw)", f"{dte_new:.0f}")
            r5, r6 = st.columns(2)
            r5.metric("σ PUT (oud → nieuw)",  f"{old_sd_put:.2f}σ → {new_sd_put:.2f}σ"  if not (np.isnan(old_sd_put) or np.isnan(new_sd_put)) else "—")
            r6.metric("σ CALL (oud → nieuw)", f"{old_sd_call:.2f}σ → {new_sd_call:.2f}σ" if not (np.isnan(old_sd_call) or np.isnan(new_sd_call)) else "—")

# ─────────────────────────── Footer ─────────────────────────────────────
st.caption("🔍 Tip: zoom met scroll/pinch, **double-click** om te rescalen. Gebruik OI/Volume-filters om spikes te filteren.")
