# pages/yield_curve.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query

st.set_page_config(page_title="Yield Curve Dashboard", layout="wide")
st.title("🧯 Yield Curve Dashboard")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})
YIELD_VIEW = TABLES.get("yield_view", f"{PROJECT_ID}.marketdata.yield_curve_dashboard_v")

# ---------- NBER recessieperioden ----------
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-30"),
]

# ---------- Kolommen ophalen ----------
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = run_query(sql, params={"tbl": tbl}, timeout=30)
    return set([c.lower() for c in dfc["column_name"].astype(str)])

cols = list_columns(YIELD_VIEW)
def have(c: str) -> bool: return c.lower() in cols

# basis yields
y2y_col = "y_2y_synth" if have("y_2y_synth") else ("y_2y" if have("y_2y") else None)
if not y2y_col:
    st.error(f"`{YIELD_VIEW}` bevat geen `y_2y_synth` of `y_2y`.")
    st.stop()

# ---------- SELECT dynamisch opbouwen ----------
select_parts = ["date"]
for src, alias in [("y_3m","y_3m"), (y2y_col,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
    if have(src): select_parts.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
if have("spread_10_2"):  select_parts.append("SAFE_CAST(spread_10_2 AS FLOAT64) AS spread_10_2")
if have("spread_30_10"): select_parts.append("SAFE_CAST(spread_30_10 AS FLOAT64) AS spread_30_10")
if have("snapshot_date"): select_parts.append("snapshot_date")

# delta-kolommen
DELTA_BASES = [
    ("y_3m","y_3m"), ("y_2y","y_2y"), ("y_5y","y_5y"),
    ("y_10y","y_10y"), ("y_30y","y_30y"),
    ("spread_10_2","spread_10_2"), ("spread_30_10","spread_30_10"),
]
HORIZONS = [("1d","_delta_bp"), ("5d","_delta_5d_bp"), ("21d","_delta_21d_bp")]
for src, alias in DELTA_BASES:
    for hname, suffix in HORIZONS:
        col = f"{src}{suffix}"
        if have(col):
            select_parts.append(f"SAFE_CAST({col} AS FLOAT64) AS {alias}{suffix}")

sql = f"SELECT {', '.join(select_parts)} FROM `{YIELD_VIEW}` ORDER BY date"

with st.spinner("Data ophalen uit BigQuery…"):
    df = run_query(sql, timeout=60)
if df.empty:
    st.warning("Geen data gevonden.")
    st.stop()

# ---------- Filters ----------
topA, topB, topC = st.columns([1.6,1,1])
with topA:
    strict = st.toggle("Strikt filter (alle looptijden aanwezig)", value=False,
                       help="Uit = alleen filteren op aanwezigheid van 2Y & 10Y.")
with topB:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with topC:
    show_table = st.toggle("Tabel tonen", value=False)

df_f = df.copy()
needed = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_f.columns]
if strict and needed:
    df_f = df_f.dropna(subset=needed)
else:
    subset = [c for c in ["y_2y","y_10y"] if c in df_f.columns]
    if subset: df_f = df_f.dropna(subset=subset)

if df_f.empty:
    st.info("Na filteren geen data over.")
    st.stop()

# ---------- Quick presets + Periode ----------
dmin = pd.to_datetime(min(df_f["date"]))
dmax = pd.to_datetime(max(df_f["date"]))
this_year_start = pd.Timestamp(year=dmax.year, month=1, day=1)

st.subheader("Periode")
left, right = st.columns([1.5, 1])
with left:
    preset = st.radio("Quick presets",
        ["Max", "YTD", "1Y", "3Y", "5Y", "10Y", "Custom"],
        horizontal=True, index=2)
with right:
    show_recessions = st.toggle("Toon US recessies (NBER)", value=True)

def clamp_start(ts: pd.Timestamp) -> pd.Timestamp:
    return max(dmin, ts)

if preset == "Max":
    start_date, end_date = dmin, dmax
elif preset == "YTD":
    start_date, end_date = clamp_start(this_year_start), dmax
elif preset == "1Y":
    start_date, end_date = clamp_start(dmax - pd.DateOffset(years=1)), dmax
elif preset == "3Y":
    start_date, end_date = clamp_start(dmax - pd.DateOffset(years=3)), dmax
elif preset == "5Y":
    start_date, end_date = clamp_start(dmax - pd.DateOffset(years=5)), dmax
elif preset == "10Y":
    start_date, end_date = clamp_start(dmax - pd.DateOffset(years=10)), dmax
else:
    date_range = st.slider("Selecteer periode (Custom)",
        min_value=dmin.to_pydatetime().date(),
        max_value=dmax.to_pydatetime().date(),
        value=(clamp_start(dmax - pd.DateOffset(years=1)).to_pydatetime().date(), dmax.to_pydatetime().date()),
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

mask = (pd.to_datetime(df_f["date"]) >= start_date) & (pd.to_datetime(df_f["date"]) <= end_date)
df_range = df_f.loc[mask].copy()
if df_range.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ---------- Snapshot-keuze ----------
st.sidebar.header("Snapshot")
all_dates = list(df_f["date"].dropna().unique())
sel_date = st.sidebar.selectbox("Kies datum", all_dates, index=len(all_dates)-1, format_func=str)
snap = df_f[df_f["date"] == sel_date].tail(1)

# ---------- KPI’s ----------
def fmt_pct(x): 
    return "—" if pd.isna(x) else f"{round(float(x), round_dp)}%"

k1, k2, k3, k4, k5 = st.columns(5)
for col, box in zip(["y_3m","y_2y","y_5y","y_10y","y_30y"], [k1,k2,k3,k4,k5]):
    val = fmt_pct(snap[col].values[0]) if (col in snap.columns and not snap.empty) else "—"
    box.metric(col.upper().replace("_",""), val)

# ---------- SIGNAL LIGHTS / ALERTS ----------
# (ongewijzigd – jouw logica voor regime/alerts hier)
# ...

# ---------- Helper recessie overlay ----------
def add_recession_shapes(fig: go.Figure, show: bool, x_start: pd.Timestamp, x_end: pd.Timestamp):
    if not show: return fig
    for start, end in NBER_RECESSIONS:
        s = pd.to_datetime(start); e = pd.to_datetime(end)
        if (e >= x_start) and (s <= x_end):
            fig.add_vrect(x0=max(s, x_start), x1=min(e, x_end),
                          fillcolor="LightGray", opacity=0.25, layer="below", line_width=0)
    return fig

# ---------- Grafieken ----------
# 1) Term Structure (snapshot)
# (ongewijzigd – jouw bestaande code)

# 2) Spreads
# (ongewijzigd – jouw bestaande code)

# 3) Rentes per looptijd (tijdreeks) + 1D Δ
st.subheader("Rentes per looptijd (tijdreeks)")

avail_yields = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
default_sel = [c for c in ["y_2y","y_10y","y_30y"] if c in avail_yields] or avail_yields[:2]
sel = st.multiselect("Selecteer looptijden", avail_yields, default=default_sel)

show_1d_delta = st.toggle("Toon 1D delta (bp) onder grafiek", value=True,
                          help="Gebruikt *_delta_bp indien aanwezig, anders diff()*100 vanaf de yield-reeks.")

if sel:
    # helper: pak bestaande *_delta_bp of bereken on-the-fly
    def get_1d_delta_bp(df_in: pd.DataFrame, col: str) -> pd.Series:
        dcol = f"{col}_delta_bp"
        if dcol in df_in.columns:
            return pd.to_numeric(df_in[dcol], errors="coerce")
        s = pd.to_numeric(df_in[col], errors="coerce")
        return s.diff() * 100.0

    fig = make_subplots(
        rows=2 if show_1d_delta else 1, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3] if show_1d_delta else [1.0],
        vertical_spacing=0.07
    )

    for col in sel:
        fig.add_trace(
            go.Scatter(x=df_range["date"], y=df_range[col], name=col.upper(), mode="lines"),
            row=1, col=1
        )

    if show_1d_delta:
        for col in sel:
            dser = get_1d_delta_bp(df_range, col)
            fig.add_trace(
                go.Bar(x=df_range["date"], y=dser, name=f"{col.upper()} Δ1D (bp)",
                       legendgroup=f"{col}_delta", showlegend=False),
                row=2, col=1
            )
        fig.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5, row=2, col=1)

    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    if show_1d_delta:
        fig.update_yaxes(title_text="Δ1D (bp)", row=2, col=1, zeroline=True)
    fig.update_xaxes(title_text="Date", row=2 if show_1d_delta else 1, col=1)

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), barmode="group")

    if show_recessions:
        x_start, x_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        for start, end in NBER_RECESSIONS:
            s = pd.to_datetime(start); e = pd.to_datetime(end)
            if (e >= x_start) and (s <= x_end):
                fig.add_vrect(
                    x0=max(s, x_start), x1=min(e, x_end),
                    fillcolor="LightGray", opacity=0.25, layer="below", line_width=0,
                    row="all", col=1
                )

    st.plotly_chart(fig, use_container_width=True)

# 4) Heatmap
# (ongewijzigd – jouw bestaande code)

# ================== DELTA-SECTIE ==================
# (ongewijzigd – jouw bestaande code)

# ---------- Tabel + download ----------
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_range.sort_values("date", ascending=False).round(round_dp))

csv = df_range.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd op periode)", data=csv,
                   file_name="yield_curve_filtered.csv", mime="text/csv")

with st.expander("Debug: kolommen in view"):
    st.write(sorted(list(cols)))
