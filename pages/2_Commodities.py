# pages/2_Commodities.py
# Works with wide view: marketdata.commodity_prices_wide_v
# Expected columns per instrument:
# <pfx>_close, <pfx>_delta_abs, <pfx>_delta_pct, <pfx>_ma20, <pfx>_ma50, <pfx>_ma200

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- BigQuery helpers ----------
try:
    from utils.bq import run_query, bq_ping
except Exception:
    import google.cloud.bigquery as bq
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bq.Client(credentials=credentials, project=creds["project_id"])

    def bq_ping() -> bool:
        try:
            _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
            return True
        except Exception:
            return False

    @st.cache_data(ttl=300, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

# ---------- Page config ----------
st.set_page_config(page_title="Commodities", layout="wide")
st.title("🛢️ Commodities Dashboard")

COM_WIDE_VIEW = st.secrets.get("tables", {}).get(
    "commodities_wide_view",
    "nth-pier-468314-p7.marketdata.commodity_prices_wide_v"
)

if not bq_ping():
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.stop()

# ---------- Load data ----------
@st.cache_data(ttl=300, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_wide = load_data()
if df_wide.empty:
    st.warning("Geen data in commodity_prices_wide_v.")
    st.stop()

# ---------- Instruments ----------
close_cols = [c for c in df_wide.columns if c.endswith("_close")]
prefixes = sorted([c[:-6] for c in close_cols])  # strip "_close"

LABELS = {
    "wti": "WTI",
    "brent": "Brent",
    "gold": "Gold",
    "silver": "Silver",
    "gasoline": "Gasoline (RBOB)",
    "heatingoil": "Heating Oil",
    "natgas": "Natural Gas",
    "copper": "Copper",
}
label_of = lambda p: LABELS.get(p, p.upper())

# ---------- UI filters ----------
min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    start, end = st.slider(
        "Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )
with c2:
    # Default: ALL instruments
    sel = st.multiselect(
        "Instrumenten (standaard alle)",
        options=[(p, label_of(p)) for p in prefixes],
        default=[(p, label_of(p)) for p in prefixes],
        format_func=lambda t: t[1],
    )
    sel = [p for p, _ in sel] or prefixes[:]
with c3:
    avg_mode = st.radio("Gemiddelde", ["EMA", "SMA"], index=0, horizontal=True)
with c4:
    show_pairs = st.checkbox("Combinatiegrafieken (Oil+Gas & Gold+Silver)", value=True)

st.caption("Elk instrument krijgt twee grafieken: links prijs + MA/EMA 20/50/200, rechts Δ% bars + YTD% (secundaire as).")

# ---------- Filtered DF ----------
mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
df = df_wide.loc[mask].sort_values("date").copy()

# ---------- Helpers ----------
def cols_for(pfx: str) -> dict:
    return {
        "close": f"{pfx}_close",
        "d_abs": f"{pfx}_delta_abs",
        "d_pct": f"{pfx}_delta_pct",
        "ma20": f"{pfx}_ma20",
        "ma50": f"{pfx}_ma50",
        "ma200": f"{pfx}_ma200",
    }

def compute_ma_ema(series: pd.Series, kind: str, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "EMA":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    return s.rolling(window=window, min_periods=window).mean()

def ytd_percent_series(dates: pd.Series, values: pd.Series) -> pd.Series:
    s = pd.Series(values.values, index=pd.to_datetime(dates), dtype="float64")
    years = s.index.year
    base = s.groupby(years).transform(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan)
    return (s / base - 1.0) * 100.0

def _to_float(s):
    return pd.to_numeric(s, errors="coerce").astype(float)

# ---------- Colors (Okabe-Ito) ----------
COLOR_PRICE   = "#111111"  # near-black
COLOR_MA20    = "#E69F00"  # orange
COLOR_MA50    = "#009E73"  # green
COLOR_MA200   = "#0072B2"  # blue
COLOR_BAR_POS = "#009E73"  # green bars
COLOR_BAR_NEG = "#D55E00"  # red bars
COLOR_YTD     = "#CC79A7"  # purple

# extra palette for combos
COLOR_WTI    = "#111111"
COLOR_BRENT  = "#0072B2"
COLOR_GAS    = "#009E73"
COLOR_GOLD   = "#E69F00"
COLOR_SILVER = "#56B4E9"

# ---------- KPIs ----------
st.subheader("Kerncijfers")
kpi_cols = st.columns(min(len(sel), 4) or 1)
for i, pfx in enumerate(sel or prefixes[:1]):
    c = cols_for(pfx)
    label = label_of(pfx)
    sub = df[["date", c["close"], c["d_abs"], c["d_pct"]]].dropna(subset=[c["close"]]).copy()
    with kpi_cols[i % len(kpi_cols)]:
        if sub.empty:
            st.metric(label, value="—", delta="—")
        else:
            last = sub.iloc[-1]
            val = float(last[c["close"]])
            d_abs = float(last[c["d_abs"]]) if pd.notnull(last[c["d_abs"]]) else 0.0
            d_pct = float(last[c["d_pct"]]) if pd.notnull(last[c["d_pct"]]) else 0.0
            st.metric(label, value=f"{val:,.2f}", delta=f"{d_abs:+.2f} ({d_pct*100:+.2f}%)")

st.markdown("---")

# ---------- Two charts per instrument ----------
st.subheader("Per instrument")
for pfx in (sel or prefixes[:1]):
    c = cols_for(pfx)
    name = label_of(pfx)
    st.markdown(f"### {name}")
    left, right = st.columns(2)

    # Left: Close + MA/EMA 20/50/200
    if c["close"] in df.columns:
        sub = df[["date", c["close"]]].dropna().copy()
        sub[c["close"]] = _to_float(sub[c["close"]])

        if avg_mode == "SMA":
            # use DB MAs if available, else compute
            ma20 = df[c["ma20"]] if c["ma20"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 20)
            ma50 = df[c["ma50"]] if c["ma50"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 50)
            ma200 = df[c["ma200"]] if c["ma200"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 200)
            ma20 = _to_float(ma20).reindex(df.index).loc[sub.index]
            ma50 = _to_float(ma50).reindex(df.index).loc[sub.index]
            ma200 = _to_float(ma200).reindex(df.index).loc[sub.index]
        else:
            ma20 = compute_ma_ema(sub[c["close"]], "EMA", 20)
            ma50 = compute_ma_ema(sub[c["close"]], "EMA", 50)
            ma200 = compute_ma_ema(sub[c["close"]], "EMA", 200)

        with left:
            fig1 = make_subplots(specs=[[{"secondary_y": False}]])
            fig1.add_trace(go.Scatter(x=sub["date"], y=sub[c["close"]], name="Close",
                                      line=dict(width=2, color=COLOR_PRICE)))
            if ma20.notna().any():
                fig1.add_trace(go.Scatter(x=sub["date"], y=ma20.values,
                                          name=("EMA20" if avg_mode=="EMA" else "MA20"),
                                          line=dict(width=2, color=COLOR_MA20)))
            if ma50.notna().any():
                fig1.add_trace(go.Scatter(x=sub["date"], y=ma50.values,
                                          name=("EMA50" if avg_mode=="EMA" else "MA50"),
                                          line=dict(width=2, color=COLOR_MA50)))
            if ma200.notna().any():
                fig1.add_trace(go.Scatter(x=sub["date"], y=ma200.values,
                                          name=("EMA200" if avg_mode=="EMA" else "MA200"),
                                          line=dict(width=2, color=COLOR_MA200)))
            fig1.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig1, use_container_width=True)
    else:
        with left:
            st.info("Geen prijsdata voor dit instrument.")

    # Right: daily Δ% bars + YTD% line
    with right:
        if c["d_pct"] not in df.columns:
            st.info("Geen dagverandering beschikbaar voor dit instrument.")
        else:
            bars = df[["date", c["d_pct"]]].dropna().copy()
            bars[c["d_pct"]] = _to_float(bars[c["d_pct"]]) * 100.0
            colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in bars[c["d_pct"]].values]

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=bars["date"], y=bars[c["d_pct"]], name="Δ% per dag",
                                  marker=dict(color=colors), opacity=0.9),
                           secondary_y=False)
            try:
                fig2.add_hline(y=0, line_dash="dot", opacity=0.5)
            except Exception:
                pass
            fig2.update_yaxes(title_text="Δ% dag", secondary_y=False)

            if c["close"] in df.columns:
                subc = df[["date", c["close"]]].dropna().copy()
                subc[c["close"]] = _to_float(subc[c["close"]])
                ytd = ytd_percent_series(subc["date"], subc[c["close"]])
                ytd = ytd.reindex(pd.to_datetime(bars["date"]))
                fig2.add_trace(go.Scatter(x=bars["date"], y=ytd.values, name="YTD%",
                                          line=dict(width=2, color=COLOR_YTD)),
                               secondary_y=True)
                fig2.update_yaxes(title_text="YTD%", secondary_y=True)

            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------- Combo charts: Oil complex & Precious metals ----------
if show_pairs:
    st.subheader("Combinatiegrafieken")

    # Energy: WTI & Brent (left) + Natural Gas (right)
    need_e = ["wti_close", "brent_close", "natgas_close"]
    if all(n in df.columns for n in need_e):
        e = df[["date"] + need_e].dropna(how="all").copy()
        e["wti_close"]    = _to_float(e["wti_close"])
        e["brent_close"]  = _to_float(e["brent_close"])
        e["natgas_close"] = _to_float(e["natgas_close"])

        fig_eng = make_subplots(specs=[[{"secondary_y": True}]])
        fig_eng.add_trace(go.Scatter(x=e["date"], y=e["wti_close"], name="WTI (USD/bbl)",
                                     line=dict(width=2, color=COLOR_WTI)), secondary_y=False)
        fig_eng.add_trace(go.Scatter(x=e["date"], y=e["brent_close"], name="Brent (USD/bbl)",
                                     line=dict(width=2, color=COLOR_BRENT)), secondary_y=False)
        fig_eng.add_trace(go.Scatter(x=e["date"], y=e["natgas_close"], name="NatGas (USD/MMBtu)",
                                     line=dict(width=2, color=COLOR_GAS)), secondary_y=True)
        fig_eng.update_yaxes(title_text="Oil price (USD/bbl)", secondary_y=False)
        fig_eng.update_yaxes(title_text="Natural Gas (USD/MMBtu)", secondary_y=True)
        fig_eng.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                              title="WTI & Brent vs Natural Gas",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig_eng, use_container_width=True)
    else:
        miss = [n for n in need_e if n not in df.columns]
        st.info(f"Energy-combo: ontbrekende kolommen: {', '.join(miss)}")

    # Metals: Gold (left) & Silver (right)
    need_m = ["gold_close", "silver_close"]
    if all(n in df.columns for n in need_m):
        m = df[["date"] + need_m].dropna(how="all").copy()
        m["gold_close"]   = _to_float(m["gold_close"])
        m["silver_close"] = _to_float(m["silver_close"])

        fig_met = make_subplots(specs=[[{"secondary_y": True}]])
        fig_met.add_trace(go.Scatter(x=m["date"], y=m["gold_close"], name="Gold (USD/oz)",
                                     line=dict(width=2, color=COLOR_GOLD)), secondary_y=False)
        fig_met.add_trace(go.Scatter(x=m["date"], y=m["silver_close"], name="Silver (USD/oz)",
                                     line=dict(width=2, color=COLOR_SILVER)), secondary_y=True)
        fig_met.update_yaxes(title_text="Gold (USD/oz)", secondary_y=False)
        fig_met.update_yaxes(title_text="Silver (USD/oz)", secondary_y=True)
        fig_met.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                              title="Gold vs Silver",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig_met, use_container_width=True)
    else:
        miss = [n for n in need_m if n not in df.columns]
        st.info(f"Metals-combo: ontbrekende kolommen: {', '.join(miss)}")

# ---------- Table ----------
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date"]
for pfx in (sel or prefixes[:1]):
    cc = cols_for(pfx)
    show_cols += [cc["close"], cc["d_abs"], cc["d_pct"], cc["ma20"], cc["ma50"], cc["ma200"]]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df[show_cols].tail(200), use_container_width=True)
