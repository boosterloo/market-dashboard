# pages/3_SPX_Options.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors as pc

from google.cloud import bigquery
from google.oauth2 import service_account

# ───────────────────────────── Small helpers ─────────────────────────────
def pick_closest_date(options: list[date], target: date):
    if not options: return None
    return min(options, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))

def pick_first_on_or_after(options: list[date], target: date):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)

def pick_closest_value(options: list[float], target: float, fallback: float | None = None):
    if not options: return fallback
    return float(min(options, key=lambda x: abs(float(x) - float(target))))

def is_round(x: float) -> int:
    # bonus: 2 voor x%100==0, 1 voor x%50==0, 1 voor x%25==0 (cumulatief)
    score = 0
    if round(x) % 100 == 0: score += 2
    if round(x) % 50 == 0:  score += 1
    if round(x) % 25 == 0:  score += 1
    return score

def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < 3: return y
    return y.rolling(window, center=True, min_periods=1).median()

# ── BigQuery client via st.secrets ────────────────────────────────
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)

_bq_client = get_bq_client()

def _bq_param(name, value):
    if isinstance(value, (list, tuple)):
        if len(value) == 0: return bigquery.ArrayQueryParameter(name, "STRING", [])
        e = value[0]
        if isinstance(e, int):   return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(e, float): return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(e, (date, pd.Timestamp, datetime)):
            return bigquery.ArrayQueryParameter(name, "DATE", [str(pd.to_datetime(v).date()) for v in value])
        return bigquery.ArrayQueryParameter(name, "STRING", [str(v) for v in value])
    if isinstance(value, bool):                 return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)):    return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)): return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime):             return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)): return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=[_bq_param(k, v) for k, v in params.items()])
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# ── Page ──────────────────────────────────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")
VIEW = "marketdata.spx_options_enriched_v"

# ── Basisfilters ──────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(f"""
        SELECT MIN(CAST(snapshot_date AS DATE)) min_date,
               MAX(CAST(snapshot_date AS DATE)) max_date
        FROM `{VIEW}`
    """)
    return df["min_date"].iloc[0], df["max_date"].iloc[0]

min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=365))

colA, colB, colC, colD, colE = st.columns([1.2, 0.8, 1, 1, 1])
with colA:
    start_date, end_date = st.date_input(
        "Periode (snapshot_date)",
        value=(default_start, max_date),
        min_value=min_date, max_value=max_date, format="YYYY-MM-DD"
    )
with colB:
    sel_type = st.radio("Type", ["call", "put"], index=1, horizontal=True)
with colC:
    dte_range = st.slider("Days to Expiration (DTE)", 0, 365, (0, 60), step=1)
with colD:
    mny_range = st.slider("Moneyness (strike/underlying − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)
with colE:
    show_underlying = st.toggle("Toon S&P500 overlay", value=True)

@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: date, end_date: date, sel_type: str):
    df = run_query(f"""
        SELECT DISTINCT expiration
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
          AND LOWER(type) = @t
        ORDER BY expiration
    """, {"start": start_date, "end": end_date, "t": sel_type})
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

exps = load_expirations(start_date, end_date, sel_type)

# ── Data laden ────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(start_date, end_date, sel_type, dte_min, dte_max, mny_min, mny_max):
    sql = f"""
    WITH base AS (
      SELECT
        snapshot_date, contract_symbol, type, expiration, days_to_exp,
        strike, underlying_price,
        SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 AS moneyness,
        (CAST(strike AS FLOAT64) - CAST(underlying_price AS FLOAT64)) AS dist_points,
        in_the_money, last_price, bid, ask, mid_price,
        implied_volatility, open_interest, volume, vix, ppd
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND LOWER(type) = @t
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0
            BETWEEN @mny_min AND @mny_max
    )
    SELECT * FROM base
    """
    params = {
        "start": start_date, "end": end_date,
        "t": sel_type,
        "dte_min": int(dte_min), "dte_max": int(dte_max),
        "mny_min": float(mny_min), "mny_max": float(mny_max),
    }
    df = run_query(sql, params=params)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"]    = pd.to_datetime(df["expiration"]).dt.date
        num_cols = ["days_to_exp","implied_volatility","open_interest","volume","ppd",
                    "strike","underlying_price","last_price","mid_price","bid","ask","dist_points"]
        for c in num_cols:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["moneyness_pct"] = 100 * df["moneyness"]
        df["abs_dist_pct"]  = (np.abs(df["dist_points"]) / df["underlying_price"]) * 100.0
        df["snap_min"] = df["snapshot_date"].dt.floor("min")
    return df

df = load_filtered(start_date, end_date, sel_type, dte_range[0], dte_range[1], mny_range[0], mny_range[1])
if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

# ── KPI's ────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Records", f"{len(df):,}")
with col2: st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with col3: st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with col4: st.metric("Som Open Interest", f"{int(df['open_interest'].sum()):,}")
st.markdown("---")

# ── Outlier-instellingen ─────────────────────────────────────────
st.caption("Outliers kunnen de schaal verstoren. Kies een methode.")
col_out1, col_out2 = st.columns([1.2, 1])
with col_out1:
    outlier_mode = st.radio(
        "Outlier-methode",
        ["Geen", "Percentiel clip", "IQR filter", "Z-score filter"],
        horizontal=True, index=1
    )
with col_out2:
    pct_clip = st.slider("Percentiel clip (links/rechts)", 0, 10, 5, step=1,
                         help="Alleen gebruikt bij 'Percentiel clip' (default 5/95).")

def apply_outlier(series: pd.Series, mode: str, pct: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if mode == "Geen":
        return s
    if mode == "Percentiel clip":
        if s.notna().any():
            lo, hi = np.nanpercentile(s, [pct, 100 - pct])
            return s.clip(lower=lo, upper=hi)
        return s
    if mode == "IQR filter":
        if s.notna().any():
            q1, q3 = np.nanpercentile(s, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return s.where((s >= lo) & (s <= hi), np.nan)
        return s
    if mode == "Z-score filter":
        mu, sd = np.nanmean(s), np.nanstd(s)
        if sd == 0 or np.isnan(sd): return s
        z = (s - mu) / sd
        return s.where(np.abs(z) <= 3.0, np.nan)
    return s

# ── Defaults: snapshot(vandaag), slimme strike, expiratie(+14d) ───────────────
snapshots_all = sorted(df["snap_min"].unique())
today_d = pd.Timestamp(date.today())

if any(pd.to_datetime(s).date() == today_d.date() for s in snapshots_all):
    default_snapshot = max([s for s in snapshots_all if pd.to_datetime(s).date() == today_d.date()])
else:
    default_snapshot = snapshots_all[-1] if snapshots_all else None

# Onderliggende op default snapshot
if default_snapshot is not None:
    sub_u = df[df["snap_min"] == default_snapshot]["underlying_price"].dropna()
    underlying_now = float(sub_u.mean()) if not sub_u.empty else (
        float(df["underlying_price"].dropna().iloc[-1]) if not df["underlying_price"].dropna().empty else np.nan
    )
else:
    underlying_now = float(df["underlying_price"].dropna().iloc[-1]) if not df["underlying_price"].dropna().empty else np.nan

# Slimme default strike (PUT −300 / CALL +200)
strikes_all = sorted([float(x) for x in df["strike"].dropna().unique().tolist()])

def choose_best_strike(df_all: pd.DataFrame, typ: str, underlying: float) -> float:
    if np.isnan(underlying) or df_all.empty: return 6000.0
    target = underlying - 300.0 if typ == "put" else underlying + 200.0
    w = 200.0  # venster half-breedte
    cand = df_all[(df_all["strike"] >= target - w) & (df_all["strike"] <= target + w)].copy()
    if cand.empty:
        return pick_closest_value(strikes_all, target, fallback=6000.0)

    grp = (cand.groupby("strike", as_index=False)
               .agg(volume=("volume","sum"),
                    oi=("open_interest","sum")))
    # normaliseer
    for c in ["volume","oi"]:
        v = grp[c].astype(float)
        if v.max() > v.min():
            grp[c+"_n"] = (v - v.min()) / (v.max() - v.min())
        else:
            grp[c+"_n"] = 0.0

    grp["dist_n"] = np.abs(grp["strike"] - target)
    if grp["dist_n"].max() > grp["dist_n"].min():
        grp["dist_n"] = (grp["dist_n"] - grp["dist_n"].min()) / (grp["dist_n"].max() - grp["dist_n"].min())
    else:
        grp["dist_n"] = 0.0

    grp["round_b"] = grp["strike"].apply(is_round).astype(float) / 4.0  # 0..1
    grp["score"] = 1.0*grp["volume_n"] + 0.5*grp["oi_n"] + 0.25*grp["round_b"] - 0.15*grp["dist_n"]
    best = grp.sort_values(["score","volume","oi"], ascending=False)["strike"].iloc[0]
    return float(best)

default_series_strike = choose_best_strike(df, sel_type, underlying_now)

# Expiratie: 2 weken vanaf vandaag
exps_all = exps
target_exp = date.today() + timedelta(days=14)
default_series_exp = pick_first_on_or_after(exps_all, target_exp) or (pick_closest_date(exps_all, target_exp) if exps_all else None)

# ── A) Serie-selectie — twee grafieken ───────────────────────────
st.subheader("Serie-selectie — volg één optiereeks door de tijd")

colS1, colS2, colS3, colS4 = st.columns([1, 1, 1, 1.6])
with colS1:
    strike_options = strikes_all or [6000.0]
    series_strike = st.selectbox(
        "Serie Strike",
        options=strike_options,
        index=(strike_options.index(default_series_strike) if default_series_strike in strike_options else 0)
    )
with colS2:
    exp_options = exps_all or [date.today()]
    series_exp = st.selectbox(
        "Serie Expiratie",
        options=exp_options,
        index=(exp_options.index(default_series_exp) if (default_series_exp in exp_options) else 0)
    )
with colS3:
    series_price_col = st.radio("Prijsbron", ["last_price","mid_price"], index=0, horizontal=True)
with colS4:
    if not np.isnan(underlying_now):
        st.caption(f"🔧 Defaults: {'PUT −300' if sel_type=='put' else 'CALL +200'} rond onderliggende ~{underlying_now:.0f} • Exp ~{target_exp}")
    else:
        st.caption("🔧 Defaults: fallback strike 6000 (onderliggende onbekend)")

serie = df[(df["strike"]==series_strike) & (df["expiration"]==series_exp)].copy().sort_values("snapshot_date")

if serie.empty:
    st.info("Geen ticks voor deze combinatie binnen de huidige filters.")
else:
    c1, c2 = st.columns(2)
    with c1:
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(go.Scatter(
            x=serie["snapshot_date"],
            y=apply_outlier(serie[series_price_col], outlier_mode, pct_clip),
            name="Price", mode="lines+markers", connectgaps=True
        ), secondary_y=False)
        if show_underlying:
            fig_price.add_trace(go.Scatter(
                x=serie["snapshot_date"], y=serie["underlying_price"],
                name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True
            ), secondary_y=True)
        fig_price.update_layout(
            title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | Price vs SP500",
            height=420, hovermode="x unified"
        )
        fig_price.update_xaxes(title_text="Meetmoment")
        fig_price.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
        fig_price.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_price, use_container_width=True)

    with c2:
        fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ppd.add_trace(go.Scatter(
            x=serie["snapshot_date"],
            y=apply_outlier(serie["ppd"], outlier_mode, pct_clip),
            name="PPD", mode="lines+markers", connectgaps=True
        ), secondary_y=False)
        if show_underlying:
            fig_ppd.add_trace(go.Scatter(
                x=serie["snapshot_date"], y=serie["underlying_price"],
                name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True
            ), secondary_y=True)
        fig_ppd.update_layout(
            title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | PPD vs SP500",
            height=420, hovermode="x unified"
        )
        fig_ppd.update_xaxes(title_text="Meetmoment")
        fig_ppd.update_yaxes(title_text="PPD", secondary_y=False, rangemode="tozero")
        fig_ppd.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_ppd, use_container_width=True)

# ── B) PPD & Afstand tot Uitoefenprijs ───────────────────────────
st.subheader("PPD & Afstand tot Uitoefenprijs (ATM→OTM/ITM)")
# default: laatste snapshot
if snapshots_all:
    default_idx = snapshots_all.index(default_snapshot) if default_snapshot in snapshots_all else len(snapshots_all)-1
else:
    default_idx = 0

sel_snapshot = st.selectbox(
    "Peildatum (snapshot)",
    options=snapshots_all, index=default_idx,
    format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
)

df_last = df[df["snap_min"] == sel_snapshot].copy()
if df_last.empty:
    near = pick_closest_date(list(snapshots_all), pd.to_datetime(sel_snapshot).date())
    if near is not None:
        df_last = df[df["snap_min"] == near].copy()

df_last["abs_dist_pct"] = ((df_last["dist_points"].abs() / df_last["underlying_price"]) * 100.0).round(3)

# binning per 0.25%-punt, medianen, min. 5 waarnemingen per bin + smoothing
bins = np.arange(0, 15.25, 0.25)
df_last["dist_bin"] = pd.cut(df_last["abs_dist_pct"], bins=bins, include_lowest=True)
g = (df_last.assign(ppd_c=apply_outlier(df_last["ppd"], outlier_mode, pct_clip))
             .groupby("dist_bin")
             .agg(ppd=("ppd_c","median"), n=("ppd_c","count"))
             .reset_index())
g = g[g["n"] >= 5].copy()
g["bin_mid"] = g["dist_bin"].apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
g = g.dropna(subset=["bin_mid"]).sort_values("bin_mid")
g["ppd_s"] = smooth_series(g["ppd"], window=3)

fig_ppd_dist = go.Figure()
fig_ppd_dist.add_trace(go.Scatter(
    x=g["bin_mid"], y=g["ppd"],
    mode="markers", name="PPD (median per bin)", opacity=0.7
))
fig_ppd_dist.add_trace(go.Scatter(
    x=g["bin_mid"], y=g["ppd_s"], mode="lines", name="Smoothed"
))
fig_ppd_dist.update_layout(
    title=f"PPD vs Afstand tot Strike — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
    xaxis_title="Afstand tot strike (|strike − underlying| / underlying, %)",
    yaxis_title="PPD (median per bin)",
    height=420
)
st.plotly_chart(fig_ppd_dist, use_container_width=True)

# ── C) Ontwikkeling Prijs per Expiratiedatum (laatste snapshot) ───────────────
st.subheader("Ontwikkeling Prijs per Expiratiedatum (laatste snapshot)")
exp_curve_raw = (
    df_last[df_last["strike"] == series_strike]
      .groupby("expiration", as_index=False)
      .agg(price=(series_price_col, "median"), ppd=("ppd", "median"))
      .sort_values("expiration")
)
exp_curve = exp_curve_raw.copy()
exp_curve["price_f"] = apply_outlier(exp_curve["price"], outlier_mode, pct_clip)
exp_curve["ppd_f"]   = apply_outlier(exp_curve["ppd"], outlier_mode, pct_clip)

fig_exp = make_subplots(specs=[[{"secondary_y": True}]])
fig_exp.add_trace(go.Scatter(
    x=exp_curve["expiration"], y=exp_curve["price_f"],
    name="Price", mode="lines+markers", connectgaps=True
), secondary_y=False)
fig_exp.add_trace(go.Scatter(
    x=exp_curve["expiration"], y=exp_curve["ppd_f"],
    name="PPD", mode="lines+markers", connectgaps=True
), secondary_y=True)
fig_exp.update_layout(
    title=f"{sel_type.upper()} — Strike {series_strike} — peildatum {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
    height=420, hovermode="x unified"
)
fig_exp.update_xaxes(title_text="Expiratiedatum")
fig_exp.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
fig_exp.update_yaxes(title_text="PPD",   secondary_y=True,  rangemode="tozero")
st.plotly_chart(fig_exp, use_container_width=True)

# ── D) PPD vs DTE ─────────────────────────────────────────────────
st.subheader("PPD vs DTE — opbouw van premium per dag")
mode_col, atm_col, win_col, base_col = st.columns([1.2, 1, 1, 1])
with mode_col:
    ppd_mode = st.radio("Bereik", ["ATM-band (moneyness)", "Rond gekozen strike"], horizontal=False, index=0)
with atm_col:
    atm_band = st.slider("ATM-band (± moneyness %)", 0.01, 0.10, 0.02, step=0.01)
with win_col:
    strike_window = st.slider("Strike-venster rond gekozen strike (punten)", 10, 200, 50, step=10)
with base_col:
    use_last_snap = st.checkbox("Gebruik alleen laatste snapshot", value=True)

base_df = df_last if use_last_snap else df

if ppd_mode.startswith("ATM"):
    df_ppd = base_df[np.abs(base_df["moneyness"]) <= atm_band].copy()
else:
    df_ppd = base_df[(base_df["strike"] >= series_strike - strike_window) &
                     (base_df["strike"] <= series_strike + strike_window)].copy()

ppd_curve = (df_ppd.assign(ppd_c=apply_outlier(df_ppd["ppd"], outlier_mode, pct_clip))
                    .groupby("days_to_exp", as_index=False)
                    .agg(ppd=("ppd_c","median"), n=("ppd_c","count"))
                    .query("n >= 5")
                    .sort_values("days_to_exp"))
ppd_curve["ppd_s"] = smooth_series(ppd_curve["ppd"], window=3)

fig_ppd_dte = go.Figure()
fig_ppd_dte.add_trace(go.Scatter(
    x=ppd_curve["days_to_exp"], y=ppd_curve["ppd"],
    mode="markers", name="PPD (median)", opacity=0.7
))
fig_ppd_dte.add_trace(go.Scatter(
    x=ppd_curve["days_to_exp"], y=ppd_curve["ppd_s"],
    mode="lines", name="Smoothed"
))
fig_ppd_dte.update_layout(
    title="PPD vs Days To Expiration",
    xaxis_title="Days to Expiration", yaxis_title="PPD (median)", height=420
)
st.plotly_chart(fig_ppd_dte, use_container_width=True)

st.markdown("---")

# ── E) Matrix — meetmoment × strike (met slimme center-zoom) ─────
st.subheader("Matrix — meetmoment × strike")
colM1, colM2, colM3, colM4 = st.columns([1, 1, 1, 1])
with colM1:
    matrix_exp = st.selectbox("Expiratie (matrix)", options=sorted(df["expiration"].unique().tolist()), index=0, key="mx_exp")
with colM2:
    matrix_metric = st.radio("Waarde", ["last_price", "mid_price", "ppd"], horizontal=False, index=0, key="mx_metric")
with colM3:
    max_rows = st.slider("Max. meetmomenten (recentste)", 50, 500, 200, step=50, key="mx_rows")
with colM4:
    center_on_best = st.checkbox("Centreer rond best-strike", value=True)
strike_window_mx = st.slider("Venster rond best-strike (punten)", 50, 400, 250, step=25, help="Toegepast wanneer centreer-optie aanstaat.")

mx = df[df["expiration"]==matrix_exp].copy().sort_values("snapshot_date").tail(max_rows)
if center_on_best and not np.isnan(underlying_now):
    best_for_mx = choose_best_strike(mx, sel_type, underlying_now)
    mx = mx[(mx["strike"] >= best_for_mx - strike_window_mx) & (mx["strike"] <= best_for_mx + strike_window_mx)]

if mx.empty:
    st.info("Geen matrix-data voor de gekozen instellingen.")
else:
    mx["snap_s"] = mx["snapshot_date"].dt.strftime("%Y-%m-%d %H:%M")
    pivot = mx.pivot_table(index="snap_s", columns="strike", values=matrix_metric, aggfunc="median")
    pivot = pivot.sort_index(ascending=False).round(2)

    arr_raw = pivot.values.astype(float)
    arr_clip = apply_outlier(pd.Series(arr_raw.flatten()), outlier_mode, pct_clip).values.reshape(arr_raw.shape)

    tab_hm, tab_tbl = st.tabs(["Heatmap", "Tabel (met kleur)"])

    with tab_hm:
        fig_mx = go.Figure(data=go.Heatmap(
            z=arr_clip,
            x=pivot.columns.astype(float),
            y=pivot.index.tolist(),
            colorbar_title=matrix_metric.capitalize(),
            hovertemplate="Snapshot: %{y}<br>Strike: %{x}<br>Value: %{z:.2f}<extra></extra>"
        ))
        title_mx = f"Heatmap — {sel_type.upper()} exp {matrix_exp} — {matrix_metric}"
        if center_on_best and not np.isnan(underlying_now):
            title_mx += f" (center rond {best_for_mx:.0f} ± {strike_window_mx})"
        fig_mx.update_layout(title=title_mx, xaxis_title="Strike", yaxis_title="Meetmoment", height=520)
        st.plotly_chart(fig_mx, use_container_width=True)

    with tab_tbl:
        scale = "Blues" if matrix_metric in ("last_price", "mid_price") else "Oranges"
        vmin = np.nanmin(arr_clip) if np.isfinite(arr_clip).any() else 0.0
        vmax = np.nanmax(arr_clip) if np.isfinite(arr_clip).any() else 1.0
        denom = (vmax - vmin) if vmax != vmin else 1.0
        norm = (np.nan_to_num(arr_clip, nan=vmin) - vmin) / denom

        cell_colors = []
        for col_idx in range(norm.shape[1]):
            col_vals = norm[:, col_idx]
            col_colors = [pc.sample_colorscale(scale, float(v)) for v in col_vals]
            cell_colors.append(col_colors)

        header_vals = ["Snapshot"] + [str(c) for c in pivot.columns.tolist()]
        cell_vals   = [pivot.index.tolist()] + [pivot[c].tolist() for c in pivot.columns.tolist()]
        header_color = pc.sample_colorscale(scale, 0.6)

        fig_tbl = go.Figure(data=[go.Table(
            header=dict(values=header_vals, fill_color=header_color, font=dict(color="white"), align="center"),
            cells=dict(values=cell_vals,
                       fill_color=[["white"]*len(pivot)] + cell_colors,
                       align="right", format=[None]+[".2f"]*len(pivot.columns))
        )])
        title_tbl = f"Tabel — {sel_type.upper()} exp {matrix_exp} — {matrix_metric}"
        if center_on_best and not np.isnan(underlying_now):
            title_tbl += f" (center rond {best_for_mx:.0f} ± {strike_window_mx})"
        fig_tbl.update_layout(title=title_tbl, height=520)
        st.plotly_chart(fig_tbl, use_container_width=True)

st.markdown("---")

# ── F) Extra: IV Term structure / IV Smile / PCR ──────────────────
term = df.groupby("days_to_exp", as_index=False)["implied_volatility"].median().sort_values("days_to_exp")
fig_term = go.Figure(go.Scatter(x=term["days_to_exp"], y=term["implied_volatility"],
                                mode="lines+markers", name=f"IV {sel_type.upper()}"))
fig_term.update_layout(title="Term Structure — Gemiddelde IV (median)", xaxis_title="DTE",
                       yaxis_title="Implied Volatility", height=380)
st.plotly_chart(fig_term, use_container_width=True)

st.subheader("IV Smile (laatste snapshot)")
exp_for_smile = st.selectbox("Expiratie voor IV Smile", options=exps_all or [None], index=0)
sm = df_last[df_last["expiration"] == exp_for_smile].copy()
if sm.empty:
    st.info("Geen data voor IV Smile.")
else:
    sm = sm.groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
    fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV"))
    fig_sm.update_layout(title=f"IV Smile — {sel_type.upper()} exp {exp_for_smile}",
                         xaxis_title="Strike", yaxis_title="Implied Volatility", height=380)
    st.plotly_chart(fig_sm, use_container_width=True)

st.subheader("Put/Call-ratio per expiratie")
p = (df.groupby(["expiration","type"], as_index=False)
       .agg(vol=("volume","sum"), oi=("open_interest","sum")))
pv = p.pivot(index="expiration", columns="type", values=["vol","oi"]).fillna(0.0)
if not pv.empty:
    pv["PCR_vol"] = pv[("vol","put")] / (pv[("vol","call")].replace(0, np.nan))
    pv["PCR_oi"]  = pv[("oi","put")]  / (pv[("oi","call")].replace(0, np.nan))
    pv = pv.replace([np.inf, -np.inf], np.nan).dropna(subset=["PCR_vol","PCR_oi"], how="all").reset_index()

    fig_pcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("PCR op Volume", "PCR op Open Interest"))
    fig_pcr.add_trace(go.Bar(x=pv["expiration"], y=pv["PCR_vol"], name="PCR (Vol)"), row=1, col=1)
    fig_pcr.add_trace(go.Bar(x=pv["expiration"], y=pv["PCR_oi"],  name="PCR (OI)"),  row=2, col=1)
    fig_pcr.update_layout(height=520, title_text="Put/Call-ratio per Expiratie")
    st.plotly_chart(fig_pcr, use_container_width=True)

# ── G) NIEUW: Volume-heatmap (strike × expiratie) ─────────────────
st.subheader("Volume-heatmap — strike × expiratie")
use_last_for_vol = st.checkbox("Gebruik alleen laatste snapshot", value=True, key="vol_last_snap")
vol_center_best  = st.checkbox("Centreer rond best-strike", value=True, key="vol_center")
vol_window_pts   = st.slider("Venster rond best-strike (punten)", 50, 500, 300, step=25, key="vol_window")

base_vol = df_last if use_last_for_vol else df
if vol_center_best and not np.isnan(underlying_now):
    best_for_vol = choose_best_strike(base_vol, sel_type, underlying_now)
    base_vol = base_vol[(base_vol["strike"] >= best_for_vol - vol_window_pts) &
                        (base_vol["strike"] <= best_for_vol + vol_window_pts)]

vol_piv = (base_vol.groupby(["expiration","strike"], as_index=False)["volume"].sum()
                 .pivot(index="expiration", columns="strike", values="volume").fillna(0.0))
if vol_piv.empty:
    st.info("Geen volume-data voor de heatmap.")
else:
    z = vol_piv.values
    fig_vol = go.Figure(data=go.Heatmap(
        z=z,
        x=vol_piv.columns.astype(float),
        y=vol_piv.index.tolist(),
        colorbar_title="Volume",
        hovertemplate="Exp: %{y}<br>Strike: %{x}<br>Vol: %{z:.0f}<extra></extra>"
    ))
    title_vol = "Volume-heatmap (strike × expiratie)"
    if vol_center_best and not np.isnan(underlying_now):
        title_vol += f" — center rond {best_for_vol:.0f} ± {vol_window_pts}"
    fig_vol.update_layout(title=title_vol, xaxis_title="Strike", yaxis_title="Expiratie", height=520)
    st.plotly_chart(fig_vol, use_container_width=True)

# VIX vs IV
vix_vs_iv = (df.assign(snap_date=df["snapshot_date"].dt.date)
               .groupby("snap_date", as_index=False)
               .agg(vix=("vix","median"), iv=("implied_volatility","median"))
               .rename(columns={"snap_date":"date"}))
if not vix_vs_iv.empty:
    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("VIX", "Gemiddelde IV"))
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["vix"], mode="lines", name="VIX"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["iv"], mode="lines", name="IV"), row=2, col=1)
    fig_vix.update_layout(height=520, title_text=f"VIX vs IV ({sel_type.upper()})")
    st.plotly_chart(fig_vix, use_container_width=True)
