# pages/4_Macro.py — Macro (Monthly, filled)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Categorieën + (optionele) dual-axis paren
GROUPS = {
    "Inflatie": ["cpi_all", "cpi_core", "pce_all"],
    "Arbeid": ["unemployment", "payrolls", "init_claims"],
    "Activiteit": ["ind_production", "retail_sales", "housing_starts"],
    "Geld & Velocity": [
        "m2", "m2_ma3", "m2_real", "m2_real_ma3",
        "m2_yoy", "m2_real_yoy", "m2_vel", "m2_vel_ma3", "m2_vel_yoy",
    ],
}
DUAL_AXIS = {
    "Inflatie": {"left": ["cpi_all"], "right": ["cpi_core", "pce_all"]},
    "Arbeid": {"left": ["unemployment"], "right": ["payrolls", "init_claims"]},
    "Activiteit": {"left": ["ind_production"], "right": ["retail_sales", "housing_starts"]},
    "Geld & Velocity": {"left": ["m2", "m2_real"], "right": ["m2_yoy", "m2_vel", "m2_vel_yoy"]},
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

    # Snelkeuze
    if st.checkbox("Selecteer alle indicatoren", value=False):
        selected = all_cols_sorted
    else:
        default_pick = [c for c in ["cpi_all", "unemployment", "ind_production"] if c in all_cols_sorted]
        if not default_pick:
            default_pick = all_cols_sorted[:4]
        selected = st.multiselect("Indicatoren", options=all_cols_sorted, default=default_pick)

    # Periode slider (default laatste 12 maanden)
    min_d, max_d = df["date"].min(), df["date"].max()
    try:
        default_start = (pd.to_datetime(max_d) - pd.DateOffset(months=12)).date()
    except Exception:
        default_start = min_d

    date_range = st.slider(
        "Periode",
        min_value=min_d,
        max_value=max_d,
        value=(max(default_start, min_d), max_d),
        step=timedelta(days=1),
        help="Schuif om de periode te kiezen. Standaard: laatste 12 maanden.",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
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

    view_mode = st.radio(
        "Weergave",
        ["Aparte grafieken (per indicator)", "Categorie-grafieken (dual-axis waar zinvol)"],
        index=1,
        help="Kies of je per indicator een grafiek wilt of per categorie met 2 y-assen.",
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

# ---------- KPI's ----------
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

# ---------- Charts ----------

def plot_single_indicator(col: str):
    series = work_tidy[["date", col]].dropna()
    if series.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["date"], y=series[col], mode="lines", name=LABELS.get(col, col)))
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
        title=LABELS.get(col, col),
        xaxis_title="Datum",
        yaxis_title=y_title,
        hovermode="x unified",
        height=420,
        legend_title="Indicator",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_group(group_name: str, cols_in_group: list[str]):
    # Bereid dual-axis set
    dual = DUAL_AXIS.get(group_name, None)
    left_list = dual.get("left", []) if dual else []
    right_list = dual.get("right", []) if dual else []

    # Filter op geselecteerde kolommen
    cols_avail = [c for c in cols_in_group if c in selected]
    if not cols_avail:
        return

    # Als geen dual-config: 1-as multi-line
    if not left_list and not right_list:
        fig = go.Figure()
        for col in cols_avail:
            s = work_tidy[["date", col]].dropna()
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=s["date"], y=s[col], mode="lines", name=LABELS.get(col, col)))
        fig.update_layout(
            title=group_name,
            xaxis_title="Datum",
            yaxis_title="Waarde",
            hovermode="x unified",
            height=460,
            legend_title="Indicator",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Dual-axis figuur
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Linkeras
    for col in [c for c in left_list if c in cols_avail]:
        s = work_tidy[["date", col]].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(x=s["date"], y=s[col], mode="lines", name=LABELS.get(col, col)),
            secondary_y=False,
        )

    # Rechteras
    for col in [c for c in right_list if c in cols_avail]:
        s = work_tidy[["date", col]].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(x=s["date"], y=s[col], mode="lines", name=LABELS.get(col, col), line=dict(dash="dash")),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"{group_name} (dual-axis)",
        hovermode="x unified",
        height=520,
        legend_title="Indicator",
    )
    fig.update_xaxes(title_text="Datum")
    fig.update_yaxes(title_text="Linker as", secondary_y=False)
    fig.update_yaxes(title_text="Rechter as", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

# Render
if view_mode == "Aparte grafieken (per indicator)":
    st.subheader("Grafieken per indicator")
    for col in selected:
        plot_single_indicator(col)
else:
    st.subheader("Grafieken per categorie")
    for gname, gcols in GROUPS.items():
        # render alleen als er overlapping is met selectie
        if any(c in selected for c in gcols):
            plot_group(gname, gcols)

# ---------- Table + download ----------
st.subheader("Data (huidige selectie)")
st.dataframe(work_tidy, use_container_width=True, hide_index=True)

csv = work_tidy.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="macro_selected.csv", mime="text/csv")

st.caption("Tip: kies 'Categorie-grafieken' om gerelateerde indicatoren naast elkaar te zien met 2 y-assen. De periode-schuiver staat standaard op de laatste 12 maanden.")
