# pages/4_Macro.py — Macro (Monthly, filled)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="📊 Macro (Monthly)", layout="wide")
st.title("📊 Macro (Monthly, filled)")

# ---------- Config ----------
DEFAULT_VIEW = "nth-pier-468314-p7.marketdata.macro_series_wide_monthly_fill_v"
MACRO_VIEW = st.secrets.get("tables", {}).get("macro_view", DEFAULT_VIEW)

LABELS = {
    "cpi_all": "CPI (headline)",
    "cpi_core": "CPI (core)",
    "pce_all": "PCE (headline)",
    "housing_starts": "Housing starts",
    "ind_production": "Industrial production",
    "init_claims": "Initial jobless claims",
    "m2": "M2",
    "m2_ma3": "M2 (3M MA)",
    "m2_real": "M2 real",
    "m2_real_ma3": "M2 real (3M MA)",
    "m2_real_yoy": "M2 real YoY %",
    "m2_vel": "M2 velocity",
    "m2_vel_ma3": "M2 velocity (3M MA)",
    "m2_vel_yoy": "M2 velocity YoY %",
    "m2_yoy": "M2 YoY %",
    "payrolls": "Payrolls (level)",
    "retail_sales": "Retail sales",
    "unemployment": "Unemployment rate",
}

# ---------- Data access ----------

def _default_run_query(sql: str) -> pd.DataFrame:
    """Fallback als utils.bq niet beschikbaar is."""
    try:
        from utils.bq import run_query  # project helper
        return run_query(sql)
    except Exception:
        try:
            from google.cloud import bigquery  # zorg dat lib geïnstalleerd is
            client = bigquery.Client.from_service_account_info(st.secrets["gcp_service_account"])  # type: ignore
            return client.query(sql).to_dataframe()
        except Exception as e:
            st.error("Geen BigQuery-verbinding. Controleer utils.bq of [gcp_service_account] in secrets.")
            st.caption(f"Details: {e}")
            return pd.DataFrame()

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_data() -> pd.DataFrame:
    sql = f"SELECT * FROM `{MACRO_VIEW}` ORDER BY date"
    df = _default_run_query(sql)
    if df.empty:
        return df
    # types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# Load
df = load_data()
if df.empty:
    st.warning("Geen data gevonden in de view.")
    st.stop()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")

    all_cols = [c for c in df.columns if c != "date"]
    all_cols_sorted = sorted(all_cols)

    default_pick = [c for c in ["cpi_all", "unemployment", "ind_production"] if c in all_cols_sorted]
    if not default_pick:
        default_pick = all_cols_sorted[:4]

    selected = st.multiselect("Indicatoren", options=all_cols_sorted, default=default_pick)

    min_d, max_d = df["date"].min(), df["date"].max()
    try:
        default_start = (pd.to_datetime(max_d) - pd.DateOffset(years=10)).date()
    except Exception:
        default_start = min_d

    date_input_val = st.date_input(
        "Periode", value=(default_start, max_d), min_value=min_d, max_value=max_d
    )
    if isinstance(date_input_val, tuple) and len(date_input_val) == 2:
        start_d, end_d = date_input_val
    else:
        start_d, end_d = min_d, max_d

    transform = st.selectbox(
        "Transformatie",
        [
            "— Geen —",
            "YoY % (12m)",
            "MoM % (1m)",
            "Index (start=100)",
            "3m MA",
            "6m MA",
            "Eerste verschillen",
        ],
        index=0,
        help="Wordt per geselecteerde serie toegepast.",
    )

    st.caption(f"Bron: `{MACRO_VIEW}`")
    if st.button("🔄 Vernieuw data (cache legen)"):
        load_data.clear()
        st.experimental_rerun()

# ---------- Filter & transform ----------
mask = (df["date"] >= start_d) & (df["date"] <= end_d)
df = df.loc[mask].copy()

if not selected:
    st.info("Selecteer minimaal één indicator.")
    st.stop()

work = df[["date"] + selected].copy().sort_values("date")
work_indexed = work.set_index("date")

# helpers

def pct_change_n(x: pd.Series, n: int) -> pd.Series:
    return x.pct_change(n) * 100.0

if transform == "YoY % (12m)":
    work_indexed = work_indexed.apply(lambda s: pct_change_n(s, 12))
elif transform == "MoM % (1m)":
    work_indexed = work_indexed.apply(lambda s: pct_change_n(s, 1))
elif transform == "Index (start=100)":
    def _to_index(s: pd.Series) -> pd.Series:
        s = s.copy()
        base = s.dropna()
        if base.empty:
            return s
        return (s / base.iloc[0]) * 100.0
    work_indexed = work_indexed.apply(_to_index)
elif transform == "3m MA":
    work_indexed = work_indexed.rolling(3, min_periods=1).mean()
elif transform == "6m MA":
    work_indexed = work_indexed.rolling(6, min_periods=1).mean()
elif transform == "Eerste verschillen":
    work_indexed = work_indexed.diff(1)

work_tidy = work_indexed.reset_index().dropna(how="all", subset=selected)

# ---------- KPI's (FIX) ----------
st.subheader("Laatste waarden")

kpi_rows = []
for col in selected:
    s = work_indexed[col].dropna()
    if s.empty:
        kpi_rows.append({"indicator": col, "value": np.nan, "prev": np.nan, "delta": np.nan})
        continue
    value = s.iloc[-1]
    prev = s.iloc[-2] if len(s) >= 2 else np.nan
    delta = value - prev if pd.notna(prev) else np.nan
    kpi_rows.append({"indicator": col, "value": value, "prev": prev, "delta": delta})

kpis = pd.DataFrame(kpi_rows).replace([np.inf, -np.inf], np.nan)

cols = st.columns(min(4, max(1, len(kpis))))
for i, row in kpis.iterrows():
    name = LABELS.get(row["indicator"], row["indicator"])
    val = row["value"]
    delta = row["delta"]
    txt_val = "—" if pd.isna(val) else f"{val:,.2f}"
    txt_delta = None if pd.isna(delta) else f"{delta:,.2f}"
    with cols[i % len(cols)]:
        st.metric(name, txt_val, delta=txt_delta)

st.divider()

# ---------- Chart ----------
fig = go.Figure()
for col in selected:
    series = work_tidy[["date", col]].dropna()
    if series.empty:
        continue
    fig.add_trace(
        go.Scatter(x=series["date"], y=series[col], mode="lines", name=LABELS.get(col, col))
    )

y_title = {
    "— Geen —": "Niveau",
    "YoY % (12m)": "YoY %",
    "MoM % (1m)": "MoM %",
    "Index (start=100)": "Index (start=100)",
    "3m MA": "Niveau (3m MA)",
    "6m MA": "Niveau (6m MA)",
    "Eerste verschillen": "Δ (m/m)",
}[transform]

fig.update_layout(
    title="Macro-reeksen (maandelijks, filled)",
    xaxis_title="Datum",
    yaxis_title=y_title,
    hovermode="x unified",
    height=480,
    legend_title="Indicator",
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Table + download ----------
st.subheader("Data (huidige selectie)")
st.dataframe(work_tidy, use_container_width=True, hide_index=True)

csv = work_tidy.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="macro_selected.csv", mime="text/csv")

st.caption("Tip: voeg meer indicatoren toe aan de onderliggende tabel; de pagina pakt ze automatisch op via de view.")
