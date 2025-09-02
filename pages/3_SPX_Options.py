# pages/3_SPX_Options.py
# ======================================================================
# 🧩 SPX Options Dashboard — auto-resolve BigQuery view names + robust UI
# ======================================================================

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, List

import plotly.graph_objects as go

# BigQuery
from google.cloud import bigquery
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="🧩 SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")

# ─────────────────────────────────────────────────────────
# Settings / Candidates
# Pas evt. je standaard project/dataset hieronder aan. Meestal komt dit
# al uit st.secrets["gcp_service_account"]["project_id"] en is dataset
# onderdeel van de view-naam. We proberen meerdere kandidaten.
# ─────────────────────────────────────────────────────────

# Je kunt deze 3 secrets optioneel zetten in .streamlit/secrets.toml:
# [bq_views]
# underlying = "project.dataset.view"
# options    = "project.dataset.view"
# snapshots  = "project.dataset.view"

VIEWS_OVERRIDE = st.secrets.get("bq_views", {})

UNDERLYING_CANDIDATES: List[str] = [
    # Exacte (vaak gebruikte) namen eerst:
    "nth-pier-468314-p7.marketdata.sp500_prices_v",
    "nth-pier-468314-p7.marketdata.spx_prices_v",
    # Kortere / alternatieve:
    "nth-pier-468314-p7.marketdata.sp500_prices",
    "nth-pier-468314-p7.marketdata.spx_prices",
    # Zonder project (als je default project al goed staat in service_account)
    "marketdata.sp500_prices_v",
    "marketdata.spx_prices_v",
    "marketdata.sp500_prices",
    "marketdata.spx_prices",
]

OPTIONS_CANDIDATES: List[str] = [
    "nth-pier-468314-p7.marketdata.spx_options_enriched_v",
    "nth-pier-468314-p7.marketdata.spx_options_v",
    "nth-pier-468314-p7.marketdata.spx_options",
    "marketdata.spx_options_enriched_v",
    "marketdata.spx_options_v",
    "marketdata.spx_options",
]

SNAPSHOT_CANDIDATES: List[str] = [
    "nth-pier-468314-p7.marketdata.spx_option_snapshots_v",
    "marketdata.spx_option_snapshots_v",
]

# UI defaults
DEFAULT_DAYS_BACK = 60
DEFAULT_SMILE_MAT_COUNT = 6
DEFAULT_SURFACE_MAT_COUNT = 8
MAX_ROWS = 300_000


# ─────────────────────────────────────────────────────────
# BigQuery helpers
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _bq_client():
    creds = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return bigquery.Client(credentials=credentials, project=creds["project_id"])

def _try_simple(sql: str) -> bool:
    """Kleine probe die alleen kijkt of de query uitvoerbaar is (LIMIT 1)."""
    try:
        _bq_client().query(sql).result(timeout=20)
        return True
    except Exception:
        return False

def run_query(sql: str) -> pd.DataFrame:
    return _bq_client().query(sql).to_dataframe(max_results=MAX_ROWS)

def _exists_view(fully_qualified: str, sample_cols: Optional[List[str]] = None) -> bool:
    """Check of view/tabel bestaat door 'SELECT <cols> FROM view LIMIT 1' te proberen."""
    cols = "*"
    if sample_cols:
        cols = ", ".join(sample_cols)
    sql = f"SELECT {cols} FROM `{fully_qualified}` LIMIT 1"
    return _try_simple(sql)

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_view(preferred: Optional[str], candidates: List[str], need_cols: Optional[List[str]] = None) -> Optional[str]:
    """Neem override uit secrets als die werkt; anders loop door candidates en pak de eerste die werkt."""
    # 1) Override vanuit secrets
    if preferred:
        if _exists_view(preferred, sample_cols=need_cols):
            return preferred
    # 2) Lijst van kandidaten
    for v in candidates:
        if _exists_view(v, sample_cols=need_cols):
            return v
    return None

@st.cache_data(ttl=60, show_spinner=False)
def bq_ping() -> bool:
    try:
        _bq_client().query("SELECT 1").result(timeout=10)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────
# Misc utils
# ─────────────────────────────────────────────────────────
def to_date(x) -> date:
    if isinstance(x, pd.Timestamp): return x.date()
    if isinstance(x, datetime):     return x.date()
    return x

def annualize_days(d: float) -> float:
    return max(float(d or 0), 0.0001) / 365.0

def safe_div(a: float, b: float) -> Optional[float]:
    try:
        return a / b if (b not in (0, None, np.nan)) else None
    except Exception:
        return None

def nearest_by_abs(df: pd.DataFrame, col: str, target: float) -> Optional[pd.Series]:
    if df.empty or col not in df.columns:
        return None
    ix = (df[col] - target).abs().idxmin()
    try:
        return df.loc[ix]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# Resolve views (once)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Verbinding")
    ok = bq_ping()
    st.success("BigQuery OK") if ok else st.error("Geen BigQuery-verbinding")

UNDERLYING_VIEW = resolve_view(
    VIEWS_OVERRIDE.get("underlying"),
    UNDERLYING_CANDIDATES,
    need_cols=["date"]
)

OPTIONS_VIEW = resolve_view(
    VIEWS_OVERRIDE.get("options"),
    OPTIONS_CANDIDATES,
    need_cols=["snapshot_date"]
)

SNAPSHOT_VIEW = resolve_view(
    VIEWS_OVERRIDE.get("snapshots"),
    SNAPSHOT_CANDIDATES,
    need_cols=["snapshot_date"]
)

with st.sidebar.expander("🔍 Debug (gevonden views)", expanded=False):
    st.write("**UNDERLYING_VIEW**:", UNDERLYING_VIEW or "❌ niet gevonden")
    st.write("**OPTIONS_VIEW**:", OPTIONS_VIEW or "❌ niet gevonden")
    st.write("**SNAPSHOT_VIEW**:", SNAPSHOT_VIEW or "❌ niet gevonden (fallback via OPTIONS_VIEW)")

if not UNDERLYING_VIEW:
    st.error("Kon geen onderliggende SPX-prijsview vinden. Pas candidates/override aan.")
    st.stop()
if not OPTIONS_VIEW:
    st.error("Kon geen SPX options-view vinden. Pas candidates/override aan.")
    st.stop()

# ─────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_underlying(days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    sql = f"""
    SELECT date, close AS spx
    FROM `{UNDERLYING_VIEW}`
    WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    ORDER BY date ASC
    """
    df = run_query(sql)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800)
def load_snapshot_dates() -> pd.DataFrame:
    # Gebruik snapshot-view als die bestaat, anders distinct uit de options-view
    if SNAPSHOT_VIEW:
        sql = f"SELECT DISTINCT snapshot_date FROM `{SNAPSHOT_VIEW}` ORDER BY snapshot_date DESC"
    else:
        sql = f"SELECT DISTINCT snapshot_date FROM `{OPTIONS_VIEW}` ORDER BY snapshot_date DESC"
    return run_query(sql)

@st.cache_data(ttl=900)
def load_options_for_snapshot(snapshot_date: date) -> pd.DataFrame:
    sql = f"""
    SELECT *
    FROM `{OPTIONS_VIEW}`
    WHERE snapshot_date = DATE('{snapshot_date}')
    """
    df = run_query(sql)
    if df.empty:
        return df

    # Normaliseer types/kolommen
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str.strip()
    for col in ["snapshot_date", "expiration"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Mid
    if "mid" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce")) / 2.0
        elif "lastPrice" in df.columns:
            df["mid"] = pd.to_numeric(df["lastPrice"], errors="coerce")
        else:
            df["mid"] = np.nan

    # DTE/TTM
    if {"expiration", "snapshot_date"}.issubset(df.columns):
        df["dte"] = (df["expiration"] - df["snapshot_date"]).dt.days.clip(lower=0)
        df["ttm"] = df["dte"].apply(annualize_days)

    # (Log) Moneyness
    if {"strike", "underlying_price"}.issubset(df.columns):
        up = pd.to_numeric(df["underlying_price"], errors="coerce")
        k  = pd.to_numeric(df["strike"], errors="coerce")
        df["moneyness"] = k / up
        df["log_moneyness"] = np.log(df["moneyness"].replace(0, np.nan))

    return df


# ─────────────────────────────────────────────────────────
# Load underlying
# ─────────────────────────────────────────────────────────
with st.spinner("Onderliggende (SPX) laden…"):
    df_spx = load_underlying(DEFAULT_DAYS_BACK)

if df_spx.empty:
    st.warning("Geen SPX-data ontvangen.")
    st.stop()

latest_date = df_spx["date"].max().date()
latest_spx = float(df_spx.loc[df_spx["date"].idxmax(), "spx"])


# ─────────────────────────────────────────────────────────
# Snapshot-datum, type, dte target
# ─────────────────────────────────────────────────────────
snap_df = load_snapshot_dates()
snap_options = [to_date(d) for d in snap_df["snapshot_date"].tolist()] if not snap_df.empty else [latest_date]
default_snap = latest_date if latest_date in snap_options else (snap_options[0] if snap_options else latest_date)

col_s1, col_s2, col_s3 = st.columns([1.2, 1, 1])
with col_s1:
    snapshot_date = st.date_input("Peildatum (snapshot_date)", value=default_snap,
                                  min_value=min(snap_options) if snap_options else latest_date,
                                  max_value=max(snap_options) if snap_options else latest_date)
with col_s2:
    put_or_call = st.radio("Type", ["PUT", "CALL"], horizontal=True, index=1)
with col_s3:
    dte_target = st.slider("Doel DTE (dagen) voor default expiratie", 7, 45, 14)

with st.spinner(f"Optie-data laden voor {snapshot_date}…"):
    df_opt = load_options_for_snapshot(snapshot_date)

if df_opt.empty:
    st.warning(f"Geen optiedata voor {snapshot_date}.")
    st.stop()

# Expiraties
exps = sorted(df_opt["expiration"].dropna().unique())
if not exps:
    st.warning("Geen expiraties gevonden voor deze snapshot.")
    st.stop()

# Kies default expiratie ~ dte_target
exp_default = min(exps, key=lambda x: abs((pd.to_datetime(x).date() - snapshot_date).days - dte_target))

col_e1, col_e2 = st.columns([1.2, 1])
with col_e1:
    expiration = st.selectbox(
        "Expiratie",
        options=[pd.to_datetime(x).date() for x in exps],
        index=[pd.to_datetime(x).date() for x in exps].index(pd.to_datetime(exp_default).date())
    )
with col_e2:
    base_S = float(df_opt["underlying_price"].dropna().iloc[0]) if "underlying_price" in df_opt.columns else latest_spx
    st.metric("SPX (onderliggende)", f"{base_S:,.2f}")


# ─────────────────────────────────────────────────────────
# Filters & slice
# ─────────────────────────────────────────────────────────
def default_strike(typ: str, S: float) -> float:
    return round(S - 500, 1) if typ.upper() == "PUT" else round(S + 300, 1)

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    strike_input = st.number_input("Strike", min_value=50.0,
                                   value=float(default_strike(put_or_call, base_S)),
                                   step=5.0, format="%.1f")
with col_k2:
    show_only_bidask = st.checkbox("Filter op zinvolle mid (bid/ask > 0)", value=True)
with col_k3:
    iv_cap = st.number_input("IV max (filter)", min_value=0.0, value=2.0, step=0.1)
with col_k4:
    oi_min = st.number_input("Min Open Interest", min_value=0, value=0, step=10)

df_exp = df_opt[df_opt["expiration"].dt.date == expiration].copy()

if show_only_bidask and {"bid", "ask"}.issubset(df_exp.columns):
    df_exp = df_exp[(pd.to_numeric(df_exp["bid"], errors="coerce") > 0) &
                    (pd.to_numeric(df_exp["ask"], errors="coerce") > 0)]

if "impliedVolatility" in df_exp.columns:
    df_exp["impliedVolatility"] = pd.to_numeric(df_exp["impliedVolatility"], errors="coerce")
    df_exp = df_exp[df_exp["impliedVolatility"].between(0, iv_cap, inclusive="both")]

if "openInterest" in df_exp.columns:
    df_exp["openInterest"] = pd.to_numeric(df_exp["openInterest"], errors="coerce").fillna(0).astype(int)
    df_exp = df_exp[df_exp["openInterest"] >= oi_min]

df_slice = df_exp[df_exp["type"] == put_or_call.upper()].copy()
if df_slice.empty:
    st.warning(f"Geen {put_or_call} contracten gevonden voor {expiration} met huidige filters.")


# ─────────────────────────────────────────────────────────
# PPD helpers
# ─────────────────────────────────────────────────────────
def premium_per_day(premium: float, dte: float) -> Optional[float]:
    if premium is None or np.isnan(premium): return None
    if dte is None or dte <= 0: return None
    return premium / dte

def mid_price(row) -> Optional[float]:
    if pd.notna(row.get("mid")): return float(row["mid"])
    if pd.notna(row.get("lastPrice")): return float(row["lastPrice"])
    if pd.notna(row.get("bid")) and pd.notna(row.get("ask")): return (row["bid"] + row["ask"]) / 2.0
    return None

def enrich_ppd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["mid_eff"] = out.apply(mid_price, axis=1)
    out["ppd"] = out.apply(lambda r: premium_per_day(r["mid_eff"], r.get("dte", None)), axis=1)
    if "underlying_price" in out.columns:
        out["ppd_pct_S"] = out.apply(lambda r: safe_div(r["ppd"], r["underlying_price"]), axis=1)
    return out

df_slice = enrich_ppd(df_slice)


# ─────────────────────────────────────────────────────────
# Charts: PPD & IV
# ─────────────────────────────────────────────────────────
st.subheader("📈 Premium per dag (PPD)")
c1, c2 = st.columns([1.5, 1])

with c1:
    fig_ppd = go.Figure()
    if not df_slice.empty:
        fig_ppd.add_trace(go.Scatter(
            x=df_slice["strike"], y=df_slice["ppd"], mode="markers+lines",
            name=f"PPD {put_or_call}",
            hovertemplate="Strike=%{x}<br>PPD=%{y:.5f}<extra></extra>"
        ))
    fig_ppd.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=30),
                          xaxis_title="Strike", yaxis_title="PPD (currency/day)")
    st.plotly_chart(fig_ppd, use_container_width=True)

with c2:
    fig_ppd_pct = go.Figure()
    if not df_slice.empty and "ppd_pct_S" in df_slice.columns:
        fig_ppd_pct.add_trace(go.Scatter(
            x=df_slice["strike"], y=df_slice["ppd_pct_S"], mode="markers",
            name="PPD / S"
        ))
    fig_ppd_pct.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=30),
                              xaxis_title="Strike", yaxis_title="PPD / S")
    st.plotly_chart(fig_ppd_pct, use_container_width=True)

st.subheader("🌀 Implied Volatility (Smile) & TTM")
c3, c4 = st.columns([1.4, 1.2])

with c3:
    fig_smile = go.Figure()
    if not df_slice.empty and "impliedVolatility" in df_slice.columns:
        fig_smile.add_trace(go.Scatter(
            x=df_slice["strike"], y=df_slice["impliedVolatility"], mode="markers+lines",
            name=f"IV {put_or_call}"
        ))
    fig_smile.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=30),
                            xaxis_title="Strike", yaxis_title="IV")
    st.plotly_chart(fig_smile, use_container_width=True)

with c4:
    df_exp_all = enrich_ppd(df_exp) if not df_exp.empty else df_exp
    fig_iv_ttm = go.Figure()
    if not df_exp_all.empty and "impliedVolatility" in df_exp_all.columns:
        fig_iv_ttm.add_trace(go.Scatter(
            x=df_exp_all["ttm"], y=df_exp_all["impliedVolatility"], mode="markers",
            name="IV vs TTM"
        ))
    fig_iv_ttm.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=30),
                             xaxis_title="TTM (jaren)", yaxis_title="IV")
    st.plotly_chart(fig_iv_ttm, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Greeks & Suggesties
# ─────────────────────────────────────────────────────────
st.subheader("🧮 Greeks & Strike-suggesties")

cand = df_slice.copy()
sug = None
if "delta" in cand.columns and not cand.empty:
    cand["abs_delta"] = cand["delta"].abs()
    # default doel |Δ| = 0.15
    sug = nearest_by_abs(cand, "abs_delta", 0.15)

info_cols = [c for c in ["type","expiration","strike","delta","gamma","theta","vega","mid_eff","ppd","openInterest"] if c in df_slice.columns]

if sug is not None:
    st.success(
        f"Suggestie {put_or_call}: "
        f"K={float(sug.get('strike')):.1f}, Δ={float(sug.get('delta', np.nan)):.3f}, "
        f"PPD={(sug.get('ppd') or 0):.5f}, mid≈{(sug.get('mid_eff') or 0):.2f}, "
        f"OI={int(sug.get('openInterest') or 0)}"
    )
else:
    st.info("Geen delta-gebaseerde suggestie beschikbaar (ontbrekende delta of lege slice).")

st.dataframe(
    df_slice[info_cols].sort_values("strike").reset_index(drop=True) if info_cols else pd.DataFrame(),
    use_container_width=True, height=300
)


# ─────────────────────────────────────────────────────────
# Smile-grid & Surface (mini heatmap)
# ─────────────────────────────────────────────────────────
st.subheader("🗺️ Smile-grid (meerdere expiraties)")

exps_dates = [pd.to_datetime(x).date() for x in exps]
exps_sorted = sorted(exps_dates, key=lambda d: abs((d - snapshot_date).days - dte_target))
pick_exps = exps_sorted[:DEFAULT_SMILE_MAT_COUNT]

cols = st.columns(len(pick_exps)) if pick_exps else []
for i, e in enumerate(pick_exps):
    with cols[i]:
        sub = df_opt[df_opt["expiration"].dt.date == e].copy()
        if show_only_bidask and {"bid", "ask"}.issubset(sub.columns):
            sub = sub[(pd.to_numeric(sub["bid"], errors="coerce") > 0) &
                      (pd.to_numeric(sub["ask"], errors="coerce") > 0)]
        if "impliedVolatility" in sub.columns:
            sub["impliedVolatility"] = pd.to_numeric(sub["impliedVolatility"], errors="coerce")
            sub = sub[sub["impliedVolatility"].between(0, iv_cap, inclusive="both")]
        sub_put = sub[sub["type"] == "PUT"] if "type" in sub.columns else pd.DataFrame()
        sub_call = sub[sub["type"] == "CALL"] if "type" in sub.columns else pd.DataFrame()

        fig = go.Figure()
        if not sub_put.empty:
            fig.add_trace(go.Scatter(x=sub_put["strike"], y=sub_put["impliedVolatility"], mode="markers+lines", name="PUT"))
        if not sub_call.empty:
            fig.add_trace(go.Scatter(x=sub_call["strike"], y=sub_call["impliedVolatility"], mode="markers+lines", name="CALL"))
        fig.update_layout(title=f"IV smile – {e}", height=300, margin=dict(l=10, r=10, t=35, b=25),
                          xaxis_title="Strike", yaxis_title="IV")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("🌋 IV Surface (mini)")
pick_surface_exps = exps_sorted[:DEFAULT_SURFACE_MAT_COUNT]
surf = df_opt[df_opt["expiration"].dt.date.isin(pick_surface_exps)].copy()
if show_only_bidask and {"bid", "ask"}.issubset(surf.columns):
    surf = surf[(pd.to_numeric(surf["bid"], errors="coerce") > 0) &
                (pd.to_numeric(surf["ask"], errors="coerce") > 0)]
if "impliedVolatility" in surf.columns:
    surf["impliedVolatility"] = pd.to_numeric(surf["impliedVolatility"], errors="coerce")
    surf = surf[surf["impliedVolatility"].between(0, iv_cap, inclusive="both")]

if not surf.empty and {"expiration","strike","impliedVolatility"}.issubset(surf.columns):
    pvt = surf.pivot_table(index="expiration", columns="strike", values="impliedVolatility", aggfunc="mean")
    pvt = pvt.sort_index().sort_index(axis=1)
    fig_hm = go.Figure(data=go.Heatmap(
        z=pvt.values,
        x=[float(x) for x in pvt.columns],
        y=[d.date() for d in pvt.index]
    ))
    fig_hm.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=30),
                         xaxis_title="Strike", yaxis_title="Expiratie",
                         title="IV heatmap (gemiddelde per strike/expiratie)")
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Onvoldoende data voor surface heatmap.")


# ─────────────────────────────────────────────────────────
# Strangle-helper
# ─────────────────────────────────────────────────────────
st.subheader("🪢 Short Strangle Helper")

col_sg1, col_sg2, col_sg3, col_sg4 = st.columns(4)
with col_sg1:
    target_delta_put = st.number_input("Target |Δ| Put", 0.05, 0.5, 0.15, step=0.05)
with col_sg2:
    target_delta_call = st.number_input("Target |Δ| Call", 0.05, 0.5, 0.10, step=0.05)
with col_sg3:
    margin_buffer_pct = st.number_input("Buffer (pct S) voor risicozones", 0.0, 0.2, 0.05, step=0.01, format="%.2f")
with col_sg4:
    show_risk_bands = st.checkbox("Toon risico-banden", value=True)

def suggest_strike(df_exp: pd.DataFrame, typ: str, target_abs_delta: float) -> Optional[pd.Series]:
    sub = df_exp[df_exp["type"] == typ.upper()].copy()
    if sub.empty or "delta" not in sub.columns:
        return None
    sub["abs_delta"] = sub["delta"].abs()
    return nearest_by_abs(sub, "abs_delta", target_abs_delta)

put_sug  = suggest_strike(df_exp, "PUT",  target_delta_put)
call_sug = suggest_strike(df_exp, "CALL", target_delta_call)

with st.expander("📋 Voorgestelde strangle details", expanded=True):
    if (put_sug is not None) and (call_sug is not None):
        S = float(df_exp["underlying_price"].dropna().iloc[0]) if "underlying_price" in df_exp.columns else base_S
        kP = float(put_sug["strike"]);  kC = float(call_sug["strike"])
        premP = float(put_sug.get("mid_eff") or 0.0); premC = float(call_sug.get("mid_eff") or 0.0)
        ppdP  = float(put_sug.get("ppd") or 0.0);     ppdC  = float(call_sug.get("ppd") or 0.0)
        width = kC - kP

        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(f"**PUT** K={kP:.1f}, Δ={float(put_sug['delta']):.3f}, PPD={ppdP:.5f}, mid≈{premP:.2f}")
        with colB:
            st.markdown(f"**CALL** K={kC:.1f}, Δ={float(call_sug['delta']):.3f}, PPD={ppdC:.5f}, mid≈{premC:.2f}")
        with colC:
            st.markdown(f"**Credit ≈** {(premP+premC):.2f}  |  **Bandbreedte** ≈ [{kP:.0f}, {kC:.0f}]  |  **Width**={width:.0f}")

        if show_risk_bands:
            low_band  = S * (1 - margin_buffer_pct)
            high_band = S * (1 + margin_buffer_pct)
            st.caption(f"Risico-banden (±{margin_buffer_pct*100:.1f}% van S={S:.1f}): {low_band:.1f} – {high_band:.1f}")
    else:
        st.info("Onvoldoende data voor strangle-suggestie (missende delta of data).")


# ─────────────────────────────────────────────────────────
# Detail gekozen strike
# ─────────────────────────────────────────────────────────
st.subheader("🔎 Detail — Gekozen contract")

choice = nearest_by_abs(df_slice, "strike", strike_input) if not df_slice.empty else None
col_d1, col_d2 = st.columns([1.6, 1.0])

with col_d1:
    fig_det = go.Figure()
    if not df_slice.empty:
        fig_det.add_trace(go.Scatter(x=df_slice["strike"], y=df_slice["mid_eff"], mode="markers+lines", name="Mid"))
        if choice is not None:
            fig_det.add_vline(x=float(choice["strike"]), line_dash="dash")
    fig_det.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=30),
                          xaxis_title="Strike", yaxis_title="Mid price")
    st.plotly_chart(fig_det, use_container_width=True)

with col_d2:
    if choice is not None:
        st.markdown("**Contract**")
        def fmt(v, f): 
            try:
                return f.format(v) if v is not None and not pd.isna(v) else "-"
            except Exception:
                return "-"
        fields = [
            ("Type", choice.get("type")),
            ("Strike", fmt(float(choice.get('strike', np.nan)), "{:.1f}")),
            ("Expiratie", to_date(choice.get("expiration")).strftime("%Y-%m-%d") if pd.notna(choice.get("expiration")) else "-"),
            ("Δ", fmt(choice.get("delta", np.nan), "{:.3f}")),
            ("Γ", fmt(choice.get("gamma", np.nan), "{:.4f}")),
            ("Θ", fmt(choice.get("theta", np.nan), "{:.3f}")),
            ("Vega", fmt(choice.get("vega", np.nan), "{:.3f}")),
            ("IV", fmt(choice.get("impliedVolatility", np.nan), "{:.3f}")),
            ("Mid", fmt(choice.get("mid_eff", 0), "{:.2f}")),
            ("PPD", fmt(choice.get("ppd", 0), "{:.5f}")),
            ("Open Interest", int(choice.get("openInterest", 0)) if pd.notna(choice.get("openInterest")) else 0),
        ]
        for k, v in fields:
            st.write(f"**{k}:** {v}")
    else:
        st.info("Geen nabijgelegen strike gevonden.")

with st.expander("📄 Raw slice-data", expanded=False):
    st.dataframe(df_slice.sort_values("strike").reset_index(drop=True), use_container_width=True, height=320)


# ─────────────────────────────────────────────────────────
# Footer / tips
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ℹ️ Als een view niet wordt gevonden, gebruik ik automatische fallbacks. "
    "Je kunt desgewenst exacte view-namen forceren via `st.secrets['bq_views']` "
    "met sleutels: `underlying`, `options`, `snapshots`."
)
