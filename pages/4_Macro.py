# pages/4_Macro.py — Macro (Monthly, filled)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote

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
DUAL_AXIS_DEFAULT = {
    "Inflatie": {"left": ["cpi_all"], "right": ["cpi_core", "pce_all"]},
    "Arbeid": {"left": ["unemployment"], "right": ["payrolls", "init_claims"]},
    "Activiteit": {"left": ["ind_production"], "right": ["retail_sales", "housing_starts"]},
    "Geld & Velocity": {"left": ["m2", "m2_real"], "right": ["m2_yoy", "m2_vel", "m2_vel_yoy"]},
}

# NBER recessies (alleen vanaf 1990 voor compactheid)
RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

# ---------- Helpers ----------

def add_recession_shading(fig: go.Figure, start_d, end_d):
    for s, e in RECESSIONS:
        s = pd.to_datetime(s).date()
        e = pd.to_datetime(e).date()
        if e < start_d or s > end_d:
            continue
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="LightGrey", opacity=0.25, line_width=0, layer="below"
        )


def y_title_for(transform: str) -> str:
    return {
        "— Geen —": "Niveau",
        "YoY % (12m)": "YoY %",
        "MoM % (1m)": "MoM %",
        "Index (start=100)": "Index (start=100)",
        "3m MA": "Niveau (3m MA)",
        "6m MA": "Niveau (6m MA)",
        "Eerste verschillen": "Δ (m/m)",
    }[transform]

# Series die in 'raw' als percentage gezien mogen worden
PCT_LEVEL_COLS = {"unemployment"}  # uitbreidbaar


def is_percent_like(transform: str) -> bool:
    return transform in {"YoY % (12m)", "MoM % (1m)"}


def percent_allowed_for_levels(transform: str) -> bool:
    """Bij deze transforms blijft de eenheid een % voor reeksen die van nature % zijn."""
    return transform in {"— Geen —", "3m MA", "6m MA", "Eerste verschillen"}


def is_percent_col(col: str) -> bool:
    """Heuristiek: kolommen die als % moeten worden getoond in raw/MA-modus."""
    c = col.lower()
    return (c in PCT_LEVEL_COLS) or c.endswith("_yoy") or ("rate" in c)


def apply_percent_axis_single(fig: go.Figure, percent_like: bool):
    if percent_like:
        try:
            fig.update_yaxes(ticksuffix="%")
        except Exception:
            pass


def apply_percent_axis_dual(fig: go.Figure, left_percent: bool, right_percent: bool):
    try:
        if left_percent:
            fig.update_yaxes(ticksuffix="%", secondary_y=False)
        if right_percent:
            fig.update_yaxes(ticksuffix="%", secondary_y=True)
    except Exception:
        pass

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

    # Periode presets + slider (default: laatste 12 maanden)
    min_d, max_d = df["date"].min(), df["date"].max()
    preset = st.radio("Preset", ["12m", "3y", "5y", "Max"], horizontal=True, index=0)
    months_map = {"12m": 12, "3y": 36, "5y": 60}
    if preset == "Max":
        default_start = min_d
    else:
        try:
            default_start = (pd.to_datetime(max_d) - pd.DateOffset(months=months_map[preset])).date()
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

    normalize = st.checkbox("Normaliseer (z-score) per serie", value=False, help="Vergelijk reeksen op dezelfde schaal.")
    shade_recessions = st.checkbox("Toon recessies (NBER)", value=True)

    view_mode = st.radio(
        "Weergave",
        ["Aparte grafieken (per indicator)", "Categorie-grafieken (dual-axis waar zinvol)"],
        index=1,
        help="Kies of je per indicator een grafiek wilt of per categorie met 2 y-assen.",
    )

    with st.expander("Geavanceerd: dual-axis per categorie aanpassen"):
        dual_custom = {}
        for gname, gcols in GROUPS.items():
        if any(c in selected for c in gcols):
            # Dual-config: haal uit sidebar-config als die bestaat, anders defaults
            try:
                dual_cfg = dual_custom.get(gname, DUAL_AXIS_DEFAULT.get(gname, {"left": [], "right": []}))
            except NameError:
                dual_cfg = DUAL_AXIS_DEFAULT.get(gname, {"left": [], "right": []})
            # Robuust plotten met foutmelding per categorie
            try:
                plot_group(gname, gcols, dual_cfg)
            except NameError as e:
                st.error(f"Kon {gname} niet plotten (NameError): {e}")
            except Exception as e:
                st.error(f"Kon {gname} niet plotten: {e}")

# ---------- Table + download ----------
st.subheader("Data (huidige selectie)")
st.dataframe(work_tidy, use_container_width=True, hide_index=True)

csv = work_tidy.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="macro_selected.csv", mime="text/csv")

st.caption("Tip: gebruik presets (12m/3y/5y/Max), recessie-shading en dual-axis per categorie voor meer context. Zet desgewenst filters in de URL voor een deelbare link.")
