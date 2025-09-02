# pages/3_SPX_Options.py
# ==============================================
# 🧩 SPX Options Dashboard
# ==============================================

# ╔═══════════════════════════════════════════╗
# 1. Imports, constants, BigQuery helpers
# ╚═══════════════════════════════════════════╝
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# BigQuery
from google.cloud import bigquery
from google.oauth2 import service_account

# Streamlit config
st.set_page_config(page_title="🧩 SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")

# Views (pas eventueel aan)
VIEW_UNDERLYING = "nth-pier-468314-p7.marketdata.sp500_prices_v"
VIEW_OPTIONS    = "nth-pier-468314-p7.marketdata.spx_options_enriched_v"
VIEW_SNAPSHOT_DATES = "nth-pier-468314-p7.marketdata.spx_option_snapshots_v"

DEFAULT_DAYS_BACK = 60
DEFAULT_SMILE_MAT_COUNT = 6
DEFAULT_SURFACE_MAT_COUNT = 8
MAX_ROWS = 300_000

# BigQuery client
@st.cache_resource(show_spinner=False)
def _bq_client():
    creds = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return bigquery.Client(credentials=credentials, project=creds["project_id"])

def run_query(sql: str) -> pd.DataFrame:
    return _bq_client().query(sql).to_dataframe(max_results=MAX_ROWS)

@st.cache_data(ttl=60, show_spinner=False)
def bq_ping() -> bool:
    try:
        _bq_client().query("SELECT 1").result(timeout=10)
        return True
    except Exception:
        return False

# Helpers
def to_date(x) -> date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    return x

def annualize_days(d: float) -> float:
    return max(d, 0.0001) / 365.0

def safe_div(a: float, b: float) -> Optional[float]:
    try:
        return a / b if b not in (0, None, np.nan) else None
    except Exception:
        return None


# ╔═══════════════════════════════════════════╗
# 2. Data loaders
# ╚═══════════════════════════════════════════╝
@st.cache_data(ttl=900)
def load_underlying(days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    sql = f"""
    SELECT date, close AS spx
    FROM `{VIEW_UNDERLYING}`
    WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    ORDER BY date ASC
    """
    df = run_query(sql)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800)
def load_snapshot_dates() -> pd.DataFrame:
    sql = f"""
    SELECT DISTINCT snapshot_date
    FROM `{VIEW_SNAPSHOT_DATES}`
    ORDER BY snapshot_date DESC
    """
    return run_query(sql)

@st.cache_data(ttl=900)
def load_options_for_snapshot(snapshot_date: date) -> pd.DataFrame:
    sql = f"""
    SELECT *
    FROM `{VIEW_OPTIONS}`
    WHERE snapshot_date = DATE('{snapshot_date}')
    """
    df = run_query(sql)
    if df.empty:
        return df
    # Normalisaties
    if "type" in df.columns:
        df["type"] = df["type"].str.upper().str.strip()
    for col in ["snapshot_date", "expiration"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "mid" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
        elif "lastPrice" in df.columns:
            df["mid"] = df["lastPrice"].astype(float)
        else:
            df["mid"] = np.nan
    if {"expiration", "snapshot_date"}.issubset(df.columns):
        df["dte"] = (df["expiration"] - df["snapshot_date"]).dt.days.clip(lower=0)
        df["ttm"] = df["dte"].apply(annualize_days)
    if {"strike", "underlying_price"}.issubset(df.columns):
        df["moneyness"] = df["strike"].astype(float) / df["underlying_price"].astype(float)
        df["log_moneyness"] = np.log(df["moneyness"].replace(0, np.nan))
    return df


# ╔═══════════════════════════════════════════╗
# 3. UI: healthcheck, snapshot, expiratie, filters
# ╚═══════════════════════════════════════════╝
with st.sidebar:
    st.markdown("### ⚙️ Verbinding")
    ok = bq_ping()
    st.success("BigQuery OK") if ok else st.error("Geen BigQuery-verbinding")

with st.spinner("Onderliggende laden…"):
    df_spx = load_underlying(DEFAULT_DAYS_BACK)

if df_spx.empty:
    st.warning("Geen SPX-data ontvangen."); st.stop()

latest_date = df_spx["date"].max().date()
latest_spx = float(df_spx.loc[df_spx["date"].idxmax(), "spx"])

snap_df = load_snapshot_dates()
snap_options = [to_date(d) for d in snap_df["snapshot_date"].tolist()] if not snap_df.empty else [latest_date]
default_snap = latest_date if latest_date in snap_options else (snap_options[0] if snap_options else latest_date)

col_s1, col_s2, col_s3 = st.columns([1.2, 1, 1])
with col_s1:
    snapshot_date = st.date_input("Peildatum", value=default_snap,
        min_value=min(snap_options) if snap_options else latest_date,
        max_value=max(snap_options) if snap_options else latest_date)
with col_s2:
    put_or_call = st.radio("Type", ["PUT", "CALL"], horizontal=True, index=1)
with col_s3:
    dte_target = st.slider("Doel DTE (dagen)", 7, 45, 14)

with st.spinner(f"Optie-data laden voor {snapshot_date}…"):
    df_opt = load_options_for_snapshot(snapshot_date)
if df_opt.empty:
    st.warning("Geen optiedata."); st.stop()

exps = sorted(df_opt["expiration"].dropna().unique())
if not exps: st.warning("Geen expiraties."); st.stop()

exp_default = min(exps, key=lambda x: abs((pd.to_datetime(x).date() - snapshot_date).days - dte_target))
col_e1, col_e2 = st.columns([1.2, 1])
with col_e1:
    expiration = st.selectbox("Expiratie",
        options=[pd.to_datetime(x).date() for x in exps],
        index=[pd.to_datetime(x).date() for x in exps].index(pd.to_datetime(exp_default).date()))
with col_e2:
    base_S = float(df_opt["underlying_price"].dropna().iloc[0]) if "underlying_price" in df_opt.columns else latest_spx
    st.metric("SPX", f"{base_S:,.2f}")

def default_strike(put_or_call: str, S: float) -> float:
    return round(S - 500, 1) if put_or_call=="PUT" else round(S + 300, 1)

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    strike_input = st.number_input("Strike", min_value=50.0,
        value=float(default_strike(put_or_call, base_S)), step=5.0, format="%.1f")
with col_k2: show_only_bidask = st.checkbox("Filter mid>0", value=True)
with col_k3: iv_cap = st.number_input("IV max", 0.0, 2.0, 2.0, 0.1)
with col_k4: oi_min = st.number_input("Min OI", 0, 0, 0, 10)

df_exp = df_opt[df_opt["expiration"].dt.date == expiration].copy()
if show_only_bidask and {"bid","ask"}.issubset(df_exp.columns):
    df_exp = df_exp[(df_exp["bid"]>0)&(df_exp["ask"]>0)]
if "impliedVolatility" in df_exp.columns:
    df_exp = df_exp[df_exp["impliedVolatility"].between(0, iv_cap)]
if "openInterest" in df_exp.columns:
    df_exp = df_exp[df_exp["openInterest"].fillna(0).astype(int) >= oi_min]
df_slice = df_exp[df_exp["type"]==put_or_call].copy()
if df_slice.empty: st.warning("Geen contracts.")


# ╔═══════════════════════════════════════════╗
# 4. PPD helpers
# ╚═══════════════════════════════════════════╝
def premium_per_day(premium: float, dte: float) -> Optional[float]:
    if premium is None or np.isnan(premium) or not dte: return None
    return premium/dte

def mid_price(row) -> Optional[float]:
    if pd.notna(row.get("mid")): return float(row["mid"])
    if pd.notna(row.get("lastPrice")): return float(row["lastPrice"])
    if pd.notna(row.get("bid")) and pd.notna(row.get("ask")): return (row["bid"]+row["ask"])/2
    return None

def enrich_ppd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mid_eff"] = df.apply(mid_price, axis=1)
    df["ppd"] = df.apply(lambda r: premium_per_day(r["mid_eff"], r.get("dte",None)), axis=1)
    if "underlying_price" in df.columns:
        df["ppd_pct_S"] = df.apply(lambda r: safe_div(r["ppd"], r["underlying_price"]), axis=1)
    return df

df_slice = enrich_ppd(df_slice)


# ╔═══════════════════════════════════════════╗
# 5. PPD grafieken + IV
# ╚═══════════════════════════════════════════╝
st.subheader("📈 PPD")
col_p1, col_p2 = st.columns([1.5,1])
with col_p1:
    fig = go.Figure()
    if not df_slice.empty:
        fig.add_trace(go.Scatter(x=df_slice["strike"], y=df_slice["ppd"], mode="markers+lines", name="PPD"))
    fig.update_layout(height=380, xaxis_title="Strike", yaxis_title="PPD")
    st.plotly_chart(fig,use_container_width=True)
with col_p2:
    fig = go.Figure()
    if "ppd_pct_S" in df_slice.columns:
        fig.add_trace(go.Scatter(x=df_slice["strike"], y=df_slice["ppd_pct_S"], mode="markers", name="PPD/S"))
    fig.update_layout(height=380, xaxis_title="Strike", yaxis_title="PPD/S")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("🌀 IV Smile & TTM")
col_iv1,col_iv2 = st.columns([1.4,1.2])
with col_iv1:
    fig = go.Figure()
    if "impliedVolatility" in df_slice.columns:
        fig.add_trace(go.Scatter(x=df_slice["strike"], y=df_slice["impliedVolatility"], mode="markers+lines"))
    fig.update_layout(height=380, xaxis_title="Strike", yaxis_title="IV")
    st.plotly_chart(fig,use_container_width=True)
with col_iv2:
    fig = go.Figure()
    df_exp_all = enrich_ppd(df_exp)
    if "impliedVolatility" in df_exp_all.columns:
        fig.add_trace(go.Scatter(x=df_exp_all["ttm"], y=df_exp_all["impliedVolatility"], mode="markers"))
    fig.update_layout(height=380, xaxis_title="TTM", yaxis_title="IV")
    st.plotly_chart(fig,use_container_width=True)


# ╔═══════════════════════════════════════════╗
# 6. Greeks & strikesuggesties
# ╚═══════════════════════════════════════════╝
st.subheader("🧮 Greeks & Suggesties")
def nearest_by_abs(df,col,target):
    if df.empty or col not in df: return None
    ix=(df[col]-target).abs().idxmin()
    return df.loc[ix]

cand=df_slice.copy()
sug=None
if "delta" in cand.columns:
    cand["abs_delta"]=cand["delta"].abs()
    sug=nearest_by_abs(cand,"abs_delta",0.15)
if sug is not None:
    st.success(f"Suggestie {put_or_call}: K={sug['strike']}, Δ={sug['delta']:.3f}, PPD={sug['ppd']:.4f}")
st.dataframe(df_slice[["type","expiration","strike","delta","gamma","theta","vega","mid_eff","ppd","openInterest"]],
             use_container_width=True,height=300)


# ╔═══════════════════════════════════════════╗
# 7. Smile-grid & Surface
# ╚═══════════════════════════════════════════╝
st.subheader("🗺️ Smile-grid")
exps_dates=[pd.to_datetime(x).date() for x in exps]
pick_exps=sorted(exps_dates,key=lambda d:abs((d-snapshot_date).days-dte_target))[:DEFAULT_SMILE_MAT_COUNT]
cols=st.columns(len(pick_exps))
for i,e in enumerate(pick_exps):
    with cols[i]:
        sub=df_opt[df_opt["expiration"].dt.date==e]
        fig=go.Figure()
        for t in ["PUT","CALL"]:
            ss=sub[sub["type"]==t]
            if not ss.empty:
                fig.add_trace(go.Scatter(x=ss["strike"], y=ss["impliedVolatility"], mode="markers+lines", name=t))
        fig.update_layout(title=f"{e}",height=300)
        st.plotly_chart(fig,use_container_width=True)

st.subheader("🌋 IV Surface")
surf=df_opt[df_opt["expiration"].dt.date.isin(exps_dates[:DEFAULT_SURFACE_MAT_COUNT])]
if not surf.empty:
    pvt=surf.pivot_table(index="expiration", columns="strike", values="impliedVolatility",aggfunc="mean")
    fig=go.Figure(data=go.Heatmap(z=pvt.values,x=[float(x) for x in pvt.columns],y=[d.date() for d in pvt.index]))
    fig.update_layout(height=420,xaxis_title="Strike",yaxis_title="Exp")
    st.plotly_chart(fig,use_container_width=True)


# ╔═══════════════════════════════════════════╗
# 8. Strangle-helper
# ╚═══════════════════════════════════════════╝
st.subheader("🪢 Strangle Helper")
def suggest_strike(df_exp,typ,target_abs_delta):
    sub=df_exp[df_exp["type"]==typ]
    if sub.empty or "delta" not in sub: return None
    sub["abs_delta"]=sub["delta"].abs()
    return nearest_by_abs(sub,"abs_delta",target_abs_delta)

put_sug=suggest_strike(df_exp,"PUT",0.15)
call_sug=suggest_strike(df_exp,"CALL",0.10)
if put_sug is not None and call_sug is not None:
    st.write(f"PUT K={put_sug['strike']} Δ={put_sug['delta']:.3f}, CALL K={call_sug['strike']} Δ={call_sug['delta']:.3f}")


# ╔═══════════════════════════════════════════╗
# 9. Detail strike
# ╚═══════════════════════════════════════════╝
st.subheader("🔎 Detail")
choice=nearest_by_abs(df_slice,"strike",strike_input) if not df_slice.empty else None
if choice is not None:
    st.write(choice)


# ╔═══════════════════════════════════════════╗
# 10. Footer
# ╚═══════════════════════════════════════════╝
st.markdown("---")
st.caption("ℹ️ Pas filters aan bij lege series. PPD is indicatief. Surface = heatmap.")
