# pages/2_Commodities.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

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

    @st.cache_data(ttl=300, show_spinner=False)
    def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

# ---- Page config ----
st.set_page_config(page_title="Commodities", layout="wide")
st.title("🛢️ Commodities Dashboard")

COM_WIDE_VIEW = st.secrets.get(
    "tables", {}
).get("commodities_wide_view", "nth-pier-468314-p7.marketdata.commodity_prices_wide_v")

# ---- Health check ----
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding.")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.caption(f"Details: {e}")
    st.stop()

# ---- Data laden ----
@st.cache_data(ttl=300, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}`")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    # 🔧 Zet alle niet-date kolommen naar float om Decimal→float issues te voorkomen
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_wide = load_data()
if df_wide.empty:
    st.warning("Geen data in commodity_prices_wide_v.")
    st.stop()

# ---- Instrument mapping (prefix -> label) ----
close_cols = [c for c in df_wide.columns if c.endswith("_close")]
prefixes = [c.removesuffix("_close") for c in close_cols]
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
prefixes = [p for p in prefixes if p in LABELS]

# ---- UI: filters ----
min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
default_start = max(min_d, max_d - timedelta(days=365)) if pd.notnull(max_d) else min_d
date_range = st.slider(
    "Periode",
    min_value=min_d,
    max_value=max_d,
    value=(default_start, max_d),
    format="YYYY-MM-DD",
)

default_selection = [p for p in ["wti", "brent", "gold", "silver"] if p in prefixes]
sel_prefixes = st.multiselect(
    "Instrumenten",
    options=[(p, LABELS[p]) for p in prefixes],
    default=[(p, LABELS[p]) for p in default_selection] or [(prefixes[0], LABELS[prefixes[0]])],
    format_func=lambda t: t[1],
)
sel_prefixes = [p for p, _ in sel_prefixes]  # haal key terug

detail_prefix = st.selectbox(
    "Detail instrument (price + MA & histogrammen)",
    options=[(p, LABELS[p]) for p in prefixes],
    index=prefixes.index(default_selection[0]) if default_selection else 0,
    format_func=lambda t: t[1],
)[0]

# ---- Filter op periode ----
mask = (df_wide["date"] >= date_range[0]) & (df_wide["date"] <= date_range[1])
df = df_wide.loc[mask].sort_values("date").copy()

def cols_for(pfx: str) -> dict:
    return {
        "close": f"{pfx}_close",
        "d_abs": f"{pfx}_delta_abs",
        "d_pct": f"{pfx}_delta_pct",
        "ma20": f"{pfx}_ma20",
        "ma50": f"{pfx}_ma50",
        "ma200": f"{pfx}_ma200",
    }

# ---- KPI's ----
st.subheader("Kerncijfers")
kpi_cols = st.columns(len(sel_prefixes) or 1)
for i, pfx in enumerate(sel_prefixes or prefixes[:1]):
    cols = cols_for(pfx)
    label = LABELS.get(pfx, pfx.upper())
    sub = df[["date", cols["close"], cols["d_abs"], cols["d_pct"]]].dropna(subset=[cols["close"]]).copy()
    if sub.empty:
        with kpi_cols[i]:
            st.metric(label, value="—", delta="—")
        continue
    last_row = sub.iloc[-1]
    val = float(last_row[cols["close"]])
    d_abs = float(last_row[cols["d_abs"]]) if pd.notnull(last_row[cols["d_abs"]]) else 0.0
    d_pct = float(last_row[cols["d_pct"]]) if pd.notnull(last_row[cols["d_pct"]]) else 0.0
    delta_str = f"{d_abs:+.2f} ({d_pct*100:+.2f}%)"
    with kpi_cols[i]:
        st.metric(label, value=f"{val:,.2f}", delta=delta_str)

st.markdown("---")

# ---- Chart 1: Normalized comparison (100 = start)
st.subheader("Vergelijking (genormaliseerd naar 100)")
fig_cmp = go.Figure()
for pfx in (sel_prefixes or prefixes[:1]):
    cols = cols_for(pfx)
    series = df[["date", cols["close"]]].dropna()
    if series.empty:
        continue
    # 🔧 robust: zet naar float en voorkom NaN/0 base
    s_close = pd.to_numeric(series[cols["close"]], errors="coerce").astype(float)
    base = s_close.dropna().iloc[0] if not s_close.dropna().empty else np.nan
    if pd.isna(base) or base == 0:
        continue
    norm = (s_close / base) * 100.0
    fig_cmp.add_trace(go.Scatter(
        x=series["date"], y=norm.values, mode="lines",
        name=LABELS.get(pfx, pfx.upper())
    ))
fig_cmp.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig_cmp, use_container_width=True)

# ---- Chart 2: Detail (close + MA20/50/200)
st.subheader(f"Detail: {LABELS.get(detail_prefix, detail_prefix.upper())} (close + MA20/50/200)")
c = cols_for(detail_prefix)
sub = df[["date", c["close"], c["ma20"], c["ma50"], c["ma200"]]].dropna(subset=[c["close"]]).copy()
for col in [c["close"], c["ma20"], c["ma50"], c["ma200"]]:
    if col in sub.columns:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").astype(float)

fig_det = go.Figure()
fig_det.add_trace(go.Scatter(x=sub["date"], y=sub[c["close"]], name="Close", mode="lines"))
if sub[c["ma20"]].notna().any(): fig_det.add_trace(go.Scatter(x=sub["date"], y=sub[c["ma20"]], name="MA20", mode="lines"))
if sub[c["ma50"]].notna().any(): fig_det.add_trace(go.Scatter(x=sub["date"], y=sub[c["ma50"]], name="MA50", mode="lines"))
if sub[c["ma200"]].notna().any(): fig_det.add_trace(go.Scatter(x=sub["date"], y=sub[c["ma200"]], name="MA200", mode="lines"))
fig_det.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_det, use_container_width=True)

# ---- Chart 3 & 4: Histogrammen delta_abs en delta_pct
left, right = st.columns(2)
with left:
    st.caption("Dagelijkse verandering (absolute delta)")
    h = df[[c["d_abs"]]].dropna()
    if h.empty:
        st.info("Geen data voor histogram.")
    else:
        s = pd.to_numeric(h[c["d_abs"]], errors="coerce").astype(float)
        fig_h1 = go.Figure()
        fig_h1.add_trace(go.Histogram(x=s))
        fig_h1.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_h1, use_container_width=True)

with right:
    st.caption("Dagelijkse verandering (procentueel)")
    h = df[[c["d_pct"]]].dropna()
    if h.empty:
        st.info("Geen data voor histogram.")
    else:
        s = pd.to_numeric(h[c["d_pct"]], errors="coerce").astype(float) * 100.0
        fig_h2 = go.Figure()
        fig_h2.add_trace(go.Histogram(x=s))
        fig_h2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_h2, use_container_width=True)

# ---- Tabel
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date"]
for pfx in (sel_prefixes or prefixes[:1]):
    cc = cols_for(pfx)
    show_cols += [cc["close"], cc["d_abs"], cc["d_pct"], cc["ma20"], cc["ma50"], cc["ma200"]]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df[show_cols].tail(200), use_container_width=True)
