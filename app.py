import os
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

# ============================
# 🎨 Design System / Defaults
# ============================
APP_TITLE = "📊 Market Dashboard"
DEFAULT_MONTHS = 4  # standaard periode = laatste 4 maanden
DATE_COL_CANDIDATES = {"date", "snapshot_date", "timestamp", "datum"}

# Plot-standaarden
PLOT_HEIGHT = 440
LEGEND_BOTTOM = True  # legenda onder de grafiek, buiten de plot
Y_MARGIN = 0.05  # 5% marge boven/onder
SHOW_ZERO_LINE = True  # referentielijn bij 0

# ============================
# ⚙️ Config & Secrets
# ============================
st.set_page_config(page_title=APP_TITLE, layout="wide")

SA_INFO = st.secrets.get("gcp_service_account", None)
PROJECT_ID = os.environ.get("PROJECT_ID") or st.secrets.get("PROJECT_ID")
DATASET = os.environ.get("DATASET") or st.secrets.get("DATASET", "marketdata")
DEFAULT_VIEW = os.environ.get("SPX_VIEW") or st.secrets.get("SPX_VIEW", "spx_analysis")

# ============================
# 🔌 BigQuery Client (cached)
# ============================
@st.cache_resource(show_spinner=False)
def get_bq_client():
    if SA_INFO:
        proj = PROJECT_ID or SA_INFO.get("project_id")
        creds = service_account.Credentials.from_service_account_info(SA_INFO)
        return bigquery.Client(credentials=creds, project=proj)
    if not PROJECT_ID:
        st.error("PROJECT_ID ontbreekt — zet deze in Secrets of als env var.")
        st.stop()
    return bigquery.Client(project=PROJECT_ID)

# ============================
# 🧰 Helpers
# ============================
@st.cache_data(ttl=300, show_spinner=False)
def list_views(dataset: str) -> List[str]:
    client = get_bq_client()
    sql = f"SELECT table_name FROM `{PROJECT_ID}.{dataset}.INFORMATION_SCHEMA.VIEWS` ORDER BY table_name"
    try:
        df = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
        return sorted(df["table_name"].tolist())
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str, params: Optional[Tuple]=None) -> pd.DataFrame:
    client = get_bq_client()
    job_config = None
    if params:
        qp = [bigquery.ScalarQueryParameter(n, t, v) for (n, t, v) in params]
        job_config = bigquery.QueryJobConfig(query_parameters=qp)
    job = client.query(sql, job_config=job_config)
    return job.result().to_dataframe(create_bqstorage_client=False)

def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    # 1) expliciete namen
    for c in df.columns:
        if c.lower() in DATE_COL_CANDIDATES:
            return c
    # 2) dtype check
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_datetime64_dtype(df[c]):
            return c
    return None

def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ===============
# 🧱 Layout UI
# ===============
with st.sidebar:
    st.title(APP_TITLE)

    # Secties (later uit te breiden)
    page = st.radio(
        "Secties",
        [
            "Time Series View",  # generieke pagina die je op elke view kunt loslaten
            "(binnenkort) Yield",
            "(binnenkort) Macro",
            "(binnenkort) Opties",
            "(binnenkort) AEX",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("🗂️ Datasource")
    _views = list_views(DATASET)
    if _views:
        selected_view = st.selectbox(
            "Kies BigQuery view",
            options=_views,
            index=_views.index(DEFAULT_VIEW) if DEFAULT_VIEW in _views else 0,
            help="Deze view wordt hieronder gevisualiseerd.",
        )
    else:
        selected_view = st.text_input(
            "View-naam", value=DEFAULT_VIEW,
            help="Kon geen views ophalen; voer handmatig een view-naam in.",
        )

    st.caption(
        f"🔐 Project: **{PROJECT_ID or '(n.v.t.)'}**  · Dataset: **{DATASET}**  · View: **{selected_view}**"
    )

# Topbar — periode over de volle breedte
col_left, col_right = st.columns([5, 1])
with col_left:
    default_end = date.today()
    default_start = default_end - timedelta(days=int(30 * DEFAULT_MONTHS))
    periode = st.date_input(
        "Periode (standaard laatste 4 maanden)",
        value=(default_start, default_end),
        help="Deze periode geldt voor de grafiek en KPI's.",
    )
with col_right:
    max_rows = st.number_input("Max. rijen", min_value=1000, max_value=500000, value=7500, step=500)

st.markdown("---")

# ===============
# 🔎 Data ophalen
# ===============
if not PROJECT_ID:
    st.stop()

table_qualified = f"`{PROJECT_ID}.{DATASET}.{selected_view}`"

# Preview om kolommen te detecteren
preview_sql = f"SELECT * FROM {table_qualified} LIMIT 50"
df_preview = run_query(preview_sql)
if df_preview.empty:
    st.warning("Geen data gevonden in de gekozen view.")
    st.stop()

# Datumkolom bepalen en query uitvoeren binnen periode
date_col = detect_date_col(df_preview)

if isinstance(periode, tuple) and len(periode) == 2 and date_col:
    start_d, end_d = periode
    sql = f"""
    SELECT *
    FROM {table_qualified}
    WHERE DATE({date_col}) BETWEEN DATE(@start) AND DATE(@end)
    ORDER BY {date_col} ASC
    LIMIT {max_rows}
    """
    params = (
        ("start", "DATE", pd.to_datetime(start_d).date()),
        ("end", "DATE", pd.to_datetime(end_d).date()),
    )
    df = run_query(sql, params=params)
else:
    st.info("Kon geen datumkolom detecteren; haal data zonder server-side filter op.")
    df = run_query(f"SELECT * FROM {table_qualified} LIMIT {max_rows}")

if df.empty:
    st.warning("Geen rijen in de geselecteerde periode.")
    st.stop()

# Zorg dat date_col typed is
if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        pass

if date_col:
    df = df.sort_values(by=date_col)

# =========================
# 📊 KPI's (consistent)
# =========================
nums = numeric_cols(df)
# heuristische 'value' kolom
preferred_value_names = ["close", "spx_close", "adj_close", "last", "value", "price"]
value_col = None
for n in preferred_value_names:
    if n in df.columns:
        value_col = n
        break
if value_col is None and nums:
    value_col = nums[0]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Rijen", f"{len(df):,}")
with k2:
    if value_col:
        st.metric(f"Laatste {value_col}", f"{df[value_col].iloc[-1]:,.2f}")
    else:
        st.metric("Numerieke kolommen", len(nums))
with k3:
    if value_col and len(df) > 1:
        delta_abs = df[value_col].iloc[-1] - df[value_col].iloc[-2]
        st.metric("Δ t.o.v. vorige", f"{delta_abs:,.2f}")
with k4:
    if value_col and len(df) > 1 and df[value_col].iloc[-2] != 0:
        delta_pct = 100 * (df[value_col].iloc[-1] / df[value_col].iloc[-2] - 1)
        st.metric("Δ %", f"{delta_pct:.2f}%")

st.markdown("---")

# =========================
# 📈 Lijngrafiek (consistent)
# =========================
# Controls
lc1, lc2, lc3 = st.columns([2,1,1])
with lc1:
    y_cols = st.multiselect(
        "Series voor grafiek",
        options=nums,
        default=[value_col] if value_col else nums[:1],
        help="Kies één of meerdere numerieke kolommen."
    )
with lc2:
    use_ma = st.toggle("Moving Average", value=True)
with lc3:
    ma_window = st.slider("MA-dagen", 5, 200, 20)

plot_df = df.copy()
if use_ma and y_cols:
    for c in y_cols:
        plot_df[f"MA_{c}"] = plot_df[c].rolling(ma_window, min_periods=1).mean()

# Build figure
if date_col and y_cols:
    plot_cols = y_cols + ([f"MA_{c}" for c in y_cols] if use_ma else [])
    fig = px.line(plot_df, x=date_col, y=plot_cols, labels={date_col: "Datum"})

    # y-as dynamisch centreren (marge + 0-lijn)
    ymin = plot_df[plot_cols].min().min()
    ymax = plot_df[plot_cols].max().max()
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    yrange = ymax - ymin
    ymin_adj = ymin - yrange * Y_MARGIN
    ymax_adj = ymax + yrange * Y_MARGIN

    fig.update_yaxes(range=[ymin_adj, ymax_adj], zeroline=SHOW_ZERO_LINE)

    # legenda onder de grafiek
    if LEGEND_BOTTOM:
        fig.update_layout(
            height=PLOT_HEIGHT,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
            margin=dict(t=10, r=10, b=10, l=10),
        )
    else:
        fig.update_layout(height=PLOT_HEIGHT)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Selecteer minimaal één numerieke serie om te plotten.")

# =========================
# 📄 Tabel (via expander)
# =========================
with st.expander("🔎 Data tabel (klik om te openen)", expanded=False):
    st.dataframe(df.tail(500), use_container_width=True)
    st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{selected_view}.csv", mime="text/csv")

# =========================
# ℹ️ Richtlijnen (voor later)
# =========================
st.markdown(
    """
**Vaste opmaakregels die nu gelden:**
- Periodebalk **bovenaan over de volle breedte** (standaard laatste 4 maanden).
- Lijngrafiek met **legenda onder de grafiek** (horizontaal), buiten de plotruimte.
- **Dynamische y-as** met marge en 0-lijn als referentie.
- **KPI-raster** met rijen, laatste waarde, Δ absoluut en Δ %.
- **Tabel** via uitklapper + **Download CSV**.
- **View-selector** in de sidebar; werkt voor alle views binnen het dataset.
    """
)
