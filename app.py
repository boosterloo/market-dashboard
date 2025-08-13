import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from google.cloud import bigquery
from google.oauth2 import service_account

# ============================
# ⚙️ Config & Secrets
# ============================
st.set_page_config(page_title="Market Dashboard", layout="wide")

# 💡 In Cloud Run gebruiken we bij voorkeur ADC (runtime service account) i.p.v. JSON-keys.
#    In Streamlit Cloud of lokaal kun je st.secrets gebruiken.
SA_INFO = st.secrets.get("gcp_service_account", None)
PROJECT_ID = os.environ.get("PROJECT_ID") or st.secrets.get("PROJECT_ID")
DATASET = os.environ.get("DATASET") or st.secrets.get("DATASET", "marketdata")
SPX_VIEW = os.environ.get("SPX_VIEW") or st.secrets.get("SPX_VIEW", "spx_analysis")

# ============================
# 🔌 BigQuery Client (cached)
# ============================
@st.cache_resource(show_spinner=False)
def get_bq_client():
    # ✅ 1) Service Account JSON via secrets (bij Streamlit Cloud of lokaal)
    if SA_INFO:
        proj = PROJECT_ID or SA_INFO.get("project_id")
        creds = service_account.Credentials.from_service_account_info(SA_INFO)
        return bigquery.Client(credentials=creds, project=proj)
    # ✅ 2) Application Default Credentials (ADC) via Cloud Run service account
    if not PROJECT_ID:
        st.error("PROJECT_ID niet gezet (env of secrets). Stel PROJECT_ID in.")
        st.stop()
    return bigquery.Client(project=PROJECT_ID)


# ============================
# 🧰 Helpers
# ============================
@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    client = get_bq_client()
    job = client.query(sql)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    return df


def detect_date_col(df: pd.DataFrame) -> str | None:
    candidates = [
        c for c in df.columns
        if c.lower() in {"date", "datum", "tradedate", "snapshot", "snapshot_date"}
    ]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_datetime64_dtype(df[c]):
            return c
    return None


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ============================
# 🎛️ Sidebar Navigation
# ============================
with st.sidebar:
    st.title("📊 Market Views")
    page = st.radio(
        "Secties",
        [
            "SPX Analysis",
            "Yield (binnenkort)",
            "Macro (binnenkort)",
            "Opties (binnenkort)",
            "AEX (binnenkort)",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("🔐 Project: **%s**  · Dataset: **%s**" % (PROJECT_ID or "(n.v.t.)", DATASET))


# ============================
# 📈 SPX Analysis Page
# ============================
if page == "SPX Analysis":
    st.title("🧠 SPX Analysis")

    # --- Filters
    col_l, col_r = st.columns([3, 1])
    with col_l:
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        start_date, end_date = st.date_input(
            "Periode",
            value=(default_start, default_end),
            help="Selecteer de periode voor de view.",
        )
    with col_r:
        limit_rows = st.number_input("Max. rijen (limiet)", min_value=200, max_value=200000, value=10000, step=500)

    table_qualified = f"`{PROJECT_ID}.{DATASET}.{SPX_VIEW}`" if PROJECT_ID else None

    if not table_qualified:
        st.error("PROJECT_ID ontbreekt — stel deze in via env var of secrets.")
        st.stop()

    # Preview om date-kolom te detecteren
    sql_preview = f"SELECT * FROM {table_qualified} LIMIT 10"
    df_preview = run_query(sql_preview)

    if df_preview.empty:
        st.warning("De view bevat (nog) geen data of is niet gevonden. Controleer project/dataset/view naam.")
        st.stop()

    date_col = detect_date_col(df_preview)

    if date_col:
        sql = f"""
        SELECT *
        FROM {table_qualified}
        WHERE DATE({date_col}) BETWEEN DATE(@start) AND DATE(@end)
        ORDER BY {date_col} ASC
        LIMIT {limit_rows}
        """
        client = get_bq_client()
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "DATE", pd.to_datetime(start_date).date()),
                bigquery.ScalarQueryParameter("end", "DATE", pd.to_datetime(end_date).date()),
            ]
        )
        df = client.query(sql, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
    else:
        st.info("Kon geen datumkolom detecteren in de view; haal data op zonder server-side filter.")
        sql = f"SELECT * FROM {table_qualified} LIMIT {limit_rows}"
        df = run_query(sql)

    if df.empty:
        st.warning("Geen rijen gevonden voor de gekozen periode.")
        st.stop()

    if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            pass

    kpi_cols = numeric_cols(df)
    if date_col:
        df = df.sort_values(by=date_col)

    preferred_names = [
        "close", "Close", "spx_close", "sp500_close", "last", "adj_close", "Adj Close"
    ]
    y_default = None
    for n in preferred_names:
        if n in df.columns:
            y_default = n
            break
    if y_default is None and kpi_cols:
        y_default = kpi_cols[0]

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Rijen", value=f"{len(df):,}")
    with kpi2:
        if y_default is not None:
            latest_val = df[y_default].iloc[-1]
            st.metric(f"Laatste {y_default}", value=f"{latest_val:,.2f}")
        else:
            st.metric("Numerieke kolommen", value=len(kpi_cols))
    with kpi3:
        if y_default is not None and len(df) > 1:
            delta_val = df[y_default].iloc[-1] - df[y_default].iloc[-2]
            st.metric("Δ t.o.v. vorige", value=f"{delta_val:,.2f}")
        else:
            st.write("")

    st.markdown("---")

    left, right = st.columns([2, 1])
    with left:
        y_col = st.selectbox("Y-as (waarde)", options=kpi_cols, index=max(0, kpi_cols.index(y_default) if y_default in kpi_cols else 0))
    with right:
        ma_window = st.slider("Moving Average (dagen)", min_value=1, max_value=200, value=20)

    df_plot = df.copy()
    if y_col:
        df_plot["MA"] = df_plot[y_col].rolling(ma_window, min_periods=1).mean()

    if date_col and y_col:
        import plotly.express as px
        fig = px.line(df_plot, x=date_col, y=[y_col, "MA"], labels={date_col: "Datum"})
        fig.update_layout(height=420, legend_title_text="Serie")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Data (laatste 200 rijen)", expanded=False):
        st.dataframe(df.tail(200), use_container_width=True)

# ============================
# 📌 Placeholder pages
# ============================
if page.startswith("Yield"):
    st.title("🧪 Yield — binnenkort")
    st.info("Deze sectie komt later: grafieken en spreads (10y-2y, 30y-10y) uit BigQuery views.")

if page.startswith("Macro"):
    st.title("🌍 Macro — binnenkort")
    st.info("Komt later: kalender, KPI's (inflatie, werkloosheid), en verrassingsindex.")

if page.startswith("Opties"):
    st.title("🧾 Opties — binnenkort")
    st.info("Komt later: SPX/AEX optie-IV, PPD en strategievergelijkingen.")

if page.startswith("AEX"):
    st.title("🇳🇱 AEX — binnenkort")
    st.info("Komt later: AEX-prijs, MA's en volatiliteitsgrafieken.")
