# pages/fx_dashboard.py
# Verwachte kolommen per pair:
# <pair>_close, <pair>_delta_abs, <pair>_delta_pct, <pair>_ma50, <pair>_ma200, optioneel <pair>_rv20, <pair>_atr14

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- BigQuery helpers ----
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

    @st.cache_data(ttl=600, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

# ---- Page config ----
st.set_page_config(page_title="🌍 FX Dashboard", layout="wide")
st.title("🌍 FX Dashboard")

FX_VIEW = st.secrets.get("tables", {}).get(
    "fx_wide_view",
    "nth-pier-468314-p7.marketdata.fx_rates_dashboard_v"
)

# ---- Health check ----
if not bq_ping():
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.stop()

# ---- Data laden ----
@st.cache_data(ttl=600, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{FX_VIEW}` ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_wide = load_data()
if df_wide.empty:
    st.warning("Geen data uit FX-view.")
    st.stop()

# ---- Pairs afleiden ----
close_cols = [c for c in df_wide.columns if c.endswith("_close")]
pairs = sorted([c.removesuffix("_close") for c in close_cols])

# Combo-sets (standaard UIT de individuele grafieken)
COMBO_GROUPS = [
    ["eurusd", "gbpusd"],
    ["usdjpy", "usdchf"],
    ["audusd", "nzdusd"],
]
COMBO_SET = [p for grp in COMBO_GROUPS for p in grp if p in pairs]

# ---- UI ----
min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    start, end = st.slider(
        "📅 Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )
with c2:
    default_sel = [p for p in pairs if p not in COMBO_SET] or pairs[:]
    sel = st.multiselect(
        "Valutaparen (individuele grafieken)",
        options=pairs,
        default=default_sel,
        format_func=lambda p: p.upper(),
    ) or default_sel
with c3:
    avg_mode = st.radio("Gemiddelde", ["EMA", "SMA"], index=0, horizontal=True)
with c4:
    show_combos = st.checkbox("Combinatiegrafieken (dual-axis)", value=True)

st.caption(
    "Standaard staan EURUSD/GBPUSD, USDJPY/USDCHF en AUDUSD/NZDUSD **alleen** in de combinatiegrafieken. "
    "Wil je ze ook individueel zien? Selecteer ze hierboven."
)

# ---- Filter ----
mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
df = df_wide.loc[mask].sort_values("date").copy()

# ---- Helpers ----
def cols_for(p: str) -> dict:
    return {
        "close": f"{p}_close",
        "d_abs": f"{p}_delta_abs",
        "d_pct": f"{p}_delta_pct",
        "ma50": f"{p}_ma50",
        "ma200": f"{p}_ma200",
        "rv20": f"{p}_rv20",
        "atr14": f"{p}_atr14",
    }

def compute_ma(series: pd.Series, kind: str, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "EMA":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    return s.rolling(window=window, min_periods=window).mean()

def ytd_percent(dates: pd.Series, values: pd.Series) -> pd.Series:
    s = pd.Series(values.values, index=pd.to_datetime(dates), dtype="float64")
    years = s.index.year
    base = s.groupby(years).transform(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan)
    return (s / base - 1.0) * 100.0

def _f(s):  # to float
    return pd.to_numeric(s, errors="coerce").astype(float)

# ---- Kleuren (Okabe-Ito) ----
COLOR_PRICE   = "#111111"  # near-black
COLOR_MA50    = "#009E73"  # green
COLOR_MA200   = "#0072B2"  # blue
COLOR_BAR_POS = "#009E73"  # green
COLOR_BAR_NEG = "#D55E00"  # red
COLOR_YTD     = "#CC79A7"  # purple

PAIR_COLOR = {
    "eurusd": "#111111",  # black
    "gbpusd": "#0072B2",  # blue
    "usdjpy": "#E69F00",  # orange
    "usdchf": "#CC79A7",  # purple
    "audusd": "#009E73",  # green
    "nzdusd": "#56B4E9",  # sky blue
}

# ---- KPI's ----
st.subheader("Kerncijfers")
kpi_cols = st.columns(min(len(sel), 4) or 1)

for i, p in enumerate(sel or pairs[:1]):
    c = cols_for(p)
    label = p.upper()
    present = ["date"] + [x for x in [c["close"], c["d_abs"], c["d_pct"]] if x in df.columns]
    sub = df[present].dropna(subset=[c["close"]]).copy() if c["close"] in df.columns else pd.DataFrame()

    with kpi_cols[i % len(kpi_cols)]:
        if sub.empty:
            st.metric(label, value="—", delta="—")
            continue
        last = sub.iloc[-1]
        val = float(last[c["close"]]) if pd.notna(last.get(c["close"])) else np.nan
        d_abs = float(last[c["d_abs"]]) if (c["d_abs"] in sub.columns and pd.notna(last.get(c["d_abs"]))) else None
        d_pct = float(last[c["d_pct"]]) if (c["d_pct"] in sub.columns and pd.notna(last.get(c["d_pct"]))) else None
        # Als delta_pct ontbreekt, bereken snel uit close:
        if d_pct is None and len(sub) >= 2:
            prev = sub.iloc[-2][c["close"]]
            if pd.notna(prev) and prev != 0:
                d_pct = (val / float(prev) - 1.0)
        delta_str = (
            f"{(d_abs if d_abs is not None else 0.0):+.5f} ({(d_pct*100 if d_pct is not None else 0.0):+.2f}%)"
        )
        st.metric(label, value=f"{val:,.5f}" if pd.notna(val) else "—", delta=delta_str)

st.markdown("---")

# ---- Per pair: 2 grafieken naast elkaar ----
st.subheader("Per pair")
for p in (sel or pairs[:1]):
    c = cols_for(p)
    st.markdown(f"### {p.upper()}")
    left, right = st.columns(2)

    # Links: Close + MA/EMA 50/200
    if c["close"] in df.columns:
        sub = df[["date", c["close"]]].dropna().copy()
        sub[c["close"]] = _f(sub[c["close"]])
        ma50 = compute_ma(sub[c["close"]], avg_mode, 50)
        ma200 = compute_ma(sub[c["close"]], avg_mode, 200)

        with left:
            fig1 = make_subplots(specs=[[{"secondary_y": False}]])
            fig1.add_trace(go.Scatter(
                x=sub["date"], y=sub[c["close"]], name="Close",
                line=dict(width=2, color=PAIR_COLOR.get(p, COLOR_PRICE))
            ))
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
            st.info("Geen close-data voor dit pair.")

    # Rechts: Δ% bars + YTD% (secundaire as)
    with right:
        # haal of bereken Δ%
        if c["d_pct"] in df.columns and df[c["d_pct"]].notna().any():
            bars = df[["date", c["d_pct"]]].dropna().copy()
            bars[c["d_pct"]] = _f(bars[c["d_pct"]]) * 100.0
        elif c["close"] in df.columns:
            tmp = df[["date", c["close"]]].dropna().copy()
            tmp[c["close"]] = _f(tmp[c["close"]])
            tmp["pct"] = tmp[c["close"]].pct_change() * 100.0
            bars = tmp[["date", "pct"]].dropna().rename(columns={"pct": c["d_pct"]})
        else:
            bars = pd.DataFrame()

        if bars.empty:
            st.info("Geen dagverandering beschikbaar voor dit pair.")
        else:
            colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in bars[c["d_pct"]].values]
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=bars["date"], y=bars[c["d_pct"]],
                                  name="Δ% per dag", marker=dict(color=colors), opacity=0.9),
                           secondary_y=False)
            try:
                fig2.add_hline(y=0, line_dash="dot", opacity=0.5)
            except Exception:
                pass
            fig2.update_yaxes(title_text="Δ% dag", secondary_y=False)

            # YTD% uit close
            if c["close"] in df.columns:
                subc = df[["date", c["close"]]].dropna().copy()
                subc[c["close"]] = _f(subc[c["close"]])
                ytd = ytd_percent(subc["date"], subc[c["close"]])
                ytd = ytd.reindex(pd.to_datetime(bars["date"]))
                fig2.add_trace(go.Scatter(x=bars["date"], y=ytd.values, name="YTD%",
                                          line=dict(width=2, color=COLOR_YTD)),
                               secondary_y=True)
                fig2.update_yaxes(title_text="YTD%", secondary_y=True)

            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---- Combinatiegrafieken (dual-axis) ----
if show_combos:
    st.subheader("Combinatiegrafieken (dual-axis)")

    def combo_chart(p_left: str, p_right: str, title: str,
                    y_left: str | None = None, y_right: str | None = None):
        cL, cR = cols_for(p_left), cols_for(p_right)
        need = []
        if cL["close"] in df.columns: need.append(cL["close"])
        if cR["close"] in df.columns: need.append(cR["close"])
        if len(need) < 2:
            st.info(f"{title}: ontbrekende kolommen.")
            return
        sub = df[["date", cL["close"], cR["close"]]].dropna(how="all").copy()
        if sub.empty:
            st.info(f"{title}: geen overlappende data.")
            return
        sub[cL["close"]] = _f(sub[cL["close"]])
        sub[cR["close"]] = _f(sub[cR["close"]])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[cL["close"]],
                                 name=p_left.upper(),
                                 line=dict(width=2, color=PAIR_COLOR.get(p_left, COLOR_PRICE))),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[cR["close"]],
                                 name=p_right.upper(),
                                 line=dict(width=2, color=PAIR_COLOR.get(p_right, COLOR_MA200))),
                      secondary_y=True)
        if y_left:  fig.update_yaxes(title_text=y_left,  secondary_y=False)
        if y_right: fig.update_yaxes(title_text=y_right, secondary_y=True)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                          title=title,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

    # EURUSD vs GBPUSD
    if all(p in pairs for p in ["eurusd", "gbpusd"]):
        combo_chart("eurusd", "gbpusd", "EURUSD vs GBPUSD",
                    y_left="EURUSD (USD per EUR)", y_right="GBPUSD (USD per GBP)")

    # USDJPY vs USDCHF
    if all(p in pairs for p in ["usdjpy", "usdchf"]):
        combo_chart("usdjpy", "usdchf", "USDJPY vs USDCHF",
                    y_left="USDJPY (JPY per USD)", y_right="USDCHF (CHF per USD)")

    # AUDUSD vs NZDUSD
    if all(p in pairs for p in ["audusd", "nzdusd"]):
        combo_chart("audusd", "nzdusd", "AUDUSD vs NZDUSD",
                    y_left="AUDUSD (USD per AUD)", y_right="NZDUSD (USD per NZD)")

# ---- Tabel (laatste 200 rijen binnen selectie) ----
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date"]
for p in (sel or pairs[:1]):
    c = cols_for(p)
    show_cols += [c["close"], c["d_abs"], c["d_pct"], c["ma50"], c["ma200"], c["rv20"], c["atr14"]]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df[show_cols].tail(200), use_container_width=True)
