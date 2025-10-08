# pages/8_Yield_Tijdreeks.py
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📈 Rentes per looptijd (tijdreeks)", layout="wide")
st.title("📈 Rentes per looptijd (tijdreeks)")

# ---- BigQuery view ----
YIELD_VIEW = st.secrets.get("tables", {}).get(
    "yield_view", "nth-pier-468314-p7.marketdata.yield_curve"
)

# ---- Healthcheck ----
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# ---- Data laden ----
@st.cache_data(ttl=1800, show_spinner=False)
def load_yields():
    q = f"""
    SELECT
      date,
      y_3m, y_2y, y_5y, y_10y, y_30y
    FROM `{YIELD_VIEW}`
    WHERE date IS NOT NULL
    ORDER BY date
    """
    df = run_query(q)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c.startswith("y_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

with st.spinner("Rentes laden…"):
    df = load_yields()
if df.empty:
    st.warning("Geen data gevonden."); st.stop()

# ---- Helpers: automatische ranges ----
def round_to_step(x: float, step: float, up: bool = True) -> float:
    if step <= 0: 
        return float(x)
    return math.ceil(x / step) * step if up else math.floor(x / step) * step

def suggest_delta_half_range_bp(df: pd.DataFrame, cols: list[str]) -> int:
    """Robuuste suggestie: 95e percentiel van |Δ1D| per serie, dan max daarvan.
       Altijd naar boven afronden op 5 bp. Min 15, max 200."""
    if not cols:
        return 25
    best = 0.0
    for c in cols:
        d = (df[c].diff() * 100.0).abs()
        val = np.nanpercentile(d.dropna(), 95) if d.notna().any() else 0.0
        best = max(best, float(val))
    best = round_to_step(best, 5.0, up=True)
    return int(min(max(best, 15), 200))

def suggest_yield_pad_pp(df: pd.DataFrame, cols: list[str]) -> float:
    """Pad in %-punt rond min/max op basis van IQR: 0.75 * IQR of 10% van span, min 0.30, max 1.50."""
    if not cols:
        return 0.60
    values = pd.concat([df[c] for c in cols], axis=0).dropna()
    if values.empty:
        return 0.60
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = max(q3 - q1, 1e-9)
    span = float(values.max() - values.min())
    pad_iqr = 0.75 * float(iqr)
    pad_span = 0.10 * float(span)
    pad = max(pad_iqr, pad_span, 0.30)
    return float(min(pad, 1.50))

# ---- UI ----
looptijd_labels = {
    "y_3m":  "3m",
    "y_2y":  "2y",
    "y_5y":  "5y",
    "y_10y": "10y",
    "y_30y": "30y",
}
beschikbaar = [c for c in looptijd_labels if c in df.columns and df[c].notna().any()]
default_selectie = [c for c in ["y_2y", "y_10y", "y_30y"] if c in beschikbaar] or beschikbaar[:3]

col_a, col_b, col_c = st.columns([2.0, 1.2, 1.0])
with col_a:
    geselecteerd = st.multiselect(
        "Selecteer looptijden",
        options=[(looptijd_labels[c], c) for c in beschikbaar],
        default=[(looptijd_labels[c], c) for c in default_selectie],
        format_func=lambda x: x[0],
    )
    geselecteerd = [c for _, c in geselecteerd]
with col_b:
    show_delta = st.toggle("Toon 1D delta (bp) onder de grafiek — één rij per looptijd", value=True)
with col_c:
    auto_scale = st.toggle("Automatisch schalen (IQR/percentiel)", value=True)

with st.sidebar:
    st.header("Periode")
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    d_range = st.date_input("Periode", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(d_range, tuple) and len(d_range) == 2:
        d_start, d_end = pd.to_datetime(d_range[0]), pd.to_datetime(d_range[1])
        m = (df["date"] >= d_start) & (df["date"] <= d_end)
        df = df.loc[m].copy()

if not geselecteerd:
    st.info("Selecteer minimaal één looptijd."); st.stop()

# Handmatige basisinstellingen (kunnen overschreven worden door auto-scale)
col1, col2, col3 = st.columns([1.3, 1.3, 1.2])
with col1:
    pad_pp = st.number_input(
        "Hoofdgrafiek marge (± %-punt)", min_value=0.0, max_value=5.0, value=0.60, step=0.05,
        help="Extra verticale ruimte in %-punt boven en onder de rentelijnen."
    )
with col2:
    delta_abs = st.number_input(
        "Delta-as half-range (bp)", min_value=5, max_value=200, value=25, step=5,
        help="Symmetrische schaal voor Δ1D-balken (bijv. 25 → ±25 bp)."
    )
with col3:
    broaden = st.button("🔎 Maak assen breder")

# ---- Automatische schalen (indien aan) ----
if auto_scale:
    pad_pp = suggest_yield_pad_pp(df, geselecteerd)     # vervangt handmatige pad
    delta_abs = suggest_delta_half_range_bp(df, geselecteerd)  # vervangt handmatige delta

# ---- Extra verbreding via knop ----
if broaden:
    pad_pp *= 1.5                 # 50% extra marge op hoofdas
    delta_abs = int(min(200, round(delta_abs * 1.5 / 5) * 5))  # 50% breder, netjes afronden op 5 bp

# ---- Δ1D (bp) berekenen ----
delta_cols = {}
for c in geselecteerd:
    dcol = f"{c}_d1bp"
    df[dcol] = df[c].diff() * 100.0
    delta_cols[c] = dcol

# ---- Subplots ----
n_delta_rows = len(geselecteerd) if show_delta else 0
rows = 1 + n_delta_rows
row_heights = [0.72] + [0.28 / max(n_delta_rows, 1)] * n_delta_rows if show_delta else [1.0]

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    specs=[[{"type": "scatter"}]] + [[{"type": "bar"}] for _ in range(n_delta_rows)],
    subplot_titles=([""] + [f"{looptijd_labels[c]} Δ1D (bp)" for c in geselecteerd] if show_delta else None)
)

# ---- Hoofdgrafiek ----
for c in geselecteerd:
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df[c],
            mode="lines", name=looptijd_labels[c],
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{looptijd_labels[c]}: %{{y:.3f}}%<extra></extra>"
        ),
        row=1, col=1
    )

ymin = float(np.nanmin([df[c].min() for c in geselecteerd]))
ymax = float(np.nanmax([df[c].max() for c in geselecteerd]))
yrange = [ymin - pad_pp, ymax + pad_pp]
fig.update_yaxes(title_text="Yield (%)", range=yrange, row=1, col=1, tickformat=".2f")

# ---- Delta-rijen ----
if show_delta:
    for i, c in enumerate(geselecteerd, start=2):
        dcol = delta_cols[c]
        pos = df[dcol].clip(lower=0)
        neg = df[dcol].clip(upper=0)
        fig.add_trace(
            go.Bar(
                x=df["date"], y=pos, name=f"{looptijd_labels[c]} +",
                marker_color="rgba(0,160,0,0.85)",
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>Δ1D: %{{y:.1f}} bp<extra></extra>",
                showlegend=False
            ),
            row=i, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df["date"], y=neg, name=f"{looptijd_labels[c]} -",
                marker_color="rgba(200,0,0,0.85)",
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>Δ1D: %{{y:.1f}} bp<extra></extra>",
                showlegend=False
            ),
            row=i, col=1
        )
        fig.update_yaxes(
            title_text=f"{looptijd_labels[c]} Δ1D (bp)",
            range=[-float(delta_abs), float(delta_abs)],
            row=i, col=1
        )

# ---- Layout ----
tot_height = 520 + (n_delta_rows * 140)
fig.update_layout(
    height=tot_height,
    margin=dict(l=40, r=20, t=40, b=40),
    barmode="relative",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
)
fig.update_xaxes(title_text="Date", row=rows, col=1)

# ---- Render ----
st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ Uitleg/instellingen"):
    st.markdown(
        f"""
- **Automatisch schalen** gebruikt robuuste statistiek:
  - Hoofdas-marge: max(**0.75×IQR**, **10% van span**), begrensd op **0.30–1.50 %-punt**.
  - Delta-half-range: **95e percentiel** van |Δ1D| over de selectie, afgerond op **5 bp**, begrensd op **15–200 bp**.
- **🔎 Maak assen breder** vergroot in één klik de hoofd-marge met **+50%** en de delta-range met **+50%**.
- Zet **Automatisch schalen** uit als je handmatig vaste waarden wil (velden boven de knop).
"""
    )
