# pages/Yield.py  — US op nieuwe enriched view
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query

st.set_page_config(page_title="Yield Curve Dashboard (US)", layout="wide")
st.title("🧯 US Yield Curve Dashboard")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})
DEFAULT_VIEW = f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v"   # <- NIEUWE VIEW
YIELD_VIEW   = TABLES.get("yield_view", DEFAULT_VIEW)

# ---------- NBER recessieperioden ----------
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-30"),
]

# ---------- Helpers ----------
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = run_query(sql, params={"tbl": tbl}, timeout=30)
    return set([c.lower() for c in dfc["column_name"].astype(str)])

def add_recession_shapes(fig: go.Figure, x_start: pd.Timestamp, x_end: pd.Timestamp, show: bool=True):
    if not show:
        return fig
    for start, end in NBER_RECESSIONS:
        s = pd.to_datetime(start); e = pd.to_datetime(end)
        if (e >= x_start) and (s <= x_end):
            fig.add_vrect(x0=max(s, x_start), x1=min(e, x_end),
                          fillcolor="LightGray", opacity=0.22, layer="below", line_width=0)
    return fig

cols = list_columns(YIELD_VIEW)
def have(c: str) -> bool: return c.lower() in cols

# basis yields (enriched views hebben y_2y_synth en/of y_2y)
y2y_pref = "y_2y_synth" if have("y_2y_synth") else ("y_2y" if have("y_2y") else None)
if not y2y_pref:
    st.error(f"`{YIELD_VIEW}` mist `y_2y_synth`/`y_2y`.")
    st.stop()

# ---------- SELECT dynamisch opbouwen ----------
select_parts = ["date"]
for src, alias in [("y_3m","y_3m"), (y2y_pref,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
    if have(src): select_parts.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
for extra in ["spread_10_2", "spread_30_10", "snapshot_date",
              "y_3m_d1_bp","y_2y_d1_bp","y_5y_d1_bp","y_10y_d1_bp","y_30y_d1_bp",
              "z_spread_10_2_252"]:
    if have(extra): select_parts.append(extra)

# 7d/30d veranderingen (pp) — converteren we in grafieken naar bp
for base in ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]:
    if have(f"{base}_d7"):  select_parts.append(f"SAFE_CAST({base}_d7  AS FLOAT64) AS {base}_d7")
    if have(f"{base}_d30"): select_parts.append(f"SAFE_CAST({base}_d30 AS FLOAT64) AS {base}_d30")

sql = f"SELECT {', '.join(select_parts)} FROM `{YIELD_VIEW}` ORDER BY date"
with st.spinner("Data ophalen uit BigQuery…"):
    df = run_query(sql, timeout=60)
if df.empty:
    st.warning("Geen data gevonden.")
    st.stop()

# ================== BOVENBALK ==================
cA, cB, cC, cD = st.columns([1.2, 1, 1, 1])
with cA:
    strict = st.toggle("Strikt (alle looptijden)", value=False,
                       help="Uit = alleen 2Y & 10Y verplicht.")
with cB:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with cC:
    show_table = st.toggle("Toon tabel", value=False)
with cD:
    show_recessions = st.toggle("US recessies (NBER)", value=True)

# Filter basis
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

# ================== PERIODE ==================
df_f["date"] = pd.to_datetime(df_f["date"])
dmin = df_f["date"].min()
dmax = df_f["date"].max()
this_year_start = pd.Timestamp(year=dmax.year, month=1, day=1)

st.subheader("Periode")
left, _ = st.columns([1.6, 1])
with left:
    preset = st.radio("Presets", ["Max", "YTD", "1Y", "3Y", "5Y", "10Y", "Custom"],
                      horizontal=True, index=2)

def clamp(ts: pd.Timestamp) -> pd.Timestamp:
    return max(dmin, ts)

if preset == "Max":
    start_date, end_date = dmin, dmax
elif preset == "YTD":
    start_date, end_date = clamp(this_year_start), dmax
elif preset == "1Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=1)), dmax
elif preset == "3Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=3)), dmax
elif preset == "5Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=5)), dmax
elif preset == "10Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=10)), dmax
else:
    date_range = st.slider(
        "Selecteer periode (Custom)",
        min_value=dmin.date(),
        max_value=dmax.date(),
        value=(clamp(dmax - pd.DateOffset(years=1)).date(), dmax.date()),
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

mask = (df_f["date"] >= start_date) & (df_f["date"] <= end_date)
df_range = df_f.loc[mask].copy()
if df_range.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ================== SNAPSHOTS ==================
st.subheader("Term Structure")

# Force Timestamps voor widgets/vergelijking
df_range["date"] = pd.to_datetime(df_range["date"])
snap_dates: list[pd.Timestamp] = sorted(df_range["date"].dropna().unique().tolist())
latest_date = pd.Timestamp(snap_dates[-1])

def nearest_date(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp:
    return min(dates, key=lambda d: abs(pd.Timestamp(d) - target)) if dates else target

one_month_prior = nearest_date(snap_dates, latest_date - pd.DateOffset(months=1))
def fmt_ts(x) -> str: return pd.Timestamp(x).strftime("%Y-%m-%d")

idx_primary = max(0, len(snap_dates) - 1)
idx_secondary = snap_dates.index(one_month_prior) if one_month_prior in snap_dates else max(0, len(snap_dates) - 2)

s1, s2, s3 = st.columns([1.2, 1.2, 1])
with s1:
    snap_primary = st.selectbox("Hoofd peildatum", options=snap_dates, index=idx_primary, format_func=fmt_ts)
with s2:
    enable_compare = st.checkbox("Vergelijk met 2e peildatum", value=True)
    snap_secondary = None
    if enable_compare:
        snap_secondary = st.selectbox("2e peildatum", options=snap_dates, index=idx_secondary, format_func=fmt_ts)
with s3:
    st.caption("Kies één of twee datums voor de curve.")

snap1 = df_range[df_range["date"] == snap_primary].tail(1)
snap2 = df_range[df_range["date"] == snap_secondary].tail(1) if snap_secondary is not None else pd.DataFrame()

# KPI’s (hoofd peildatum)
def fmt_pct(x): 
    return "—" if pd.isna(x) else f"{round(float(x), round_dp)}%"

k1, k2, k3, k4, k5 = st.columns(5)
for col, box in zip(["y_3m","y_2y","y_5y","y_10y","y_30y"], [k1,k2,k3,k4,k5]):
    val = fmt_pct(snap1[col].values[0]) if (col in snap1.columns and not snap1.empty) else "—"
    box.metric(col.upper().replace("_",""), val)

# Term-structure plot (met optionele 2e curve)
def curve_points(row: pd.Series):
    maturities = ["3M", "2Y", "5Y", "10Y", "30Y"]
    vals = [row.get("y_3m"), row.get("y_2y"), row.get("y_5y"), row.get("y_10y"), row.get("y_30y")]
    m = [m for m, v in zip(maturities, vals) if pd.notna(v)]
    v = [v for v in vals if pd.notna(v)]
    return m, v

ts_fig = go.Figure()
if not snap1.empty:
    m, v = curve_points(snap1.iloc[0])
    ts_fig.add_trace(go.Scatter(x=m, y=v, mode="lines+markers", name=str(pd.Timestamp(snap_primary).date())))
if enable_compare and not snap2.empty:
    m2, v2 = curve_points(snap2.iloc[0])
    ts_fig.add_trace(go.Scatter(x=m2, y=v2, mode="lines+markers",
                                name=str(pd.Timestamp(snap_secondary).date()), line=dict(dash="dash")))
ts_fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Maturity")
st.plotly_chart(ts_fig, use_container_width=True)
st.caption("Normaal = stijgend; vlak = late cyclus; invers = recessierisico/verwachte cuts.")

# ================== SIGNALS ==================
st.subheader("Signals")
sigL, sigR = st.columns([1.3, 1])
with sigR:
    # enriched view: 7d / 30d (waarden in procentpunten)
    regime_horizon = st.radio("Regime-horizon", ["7d", "30d"], horizontal=True, index=0)
    suffix = "_d7" if regime_horizon == "7d" else "_d30"
    default_steepen = 10 if regime_horizon == "7d" else 15
    default_bigmove = 12 if regime_horizon == "7d" else 18
    st.caption(f"Drempels ({regime_horizon} Δ in basispunten)")
    thr_steepen = st.slider("Steil/plat drempel |Δ(10Y–2Y)|", 5, 60, default_steepen, step=1)
    thr_bigmove  = st.slider("Grote move per reeks", 5, 60, default_bigmove, step=1)

def last_val(col: str):
    if col not in df_range.columns: return None
    s = pd.to_numeric(df_range[col], errors="coerce").dropna()
    return None if s.empty else float(s.iloc[-1])

# Δ(10Y–2Y) op gekozen horizon: pp → bp
d_spread = None
if f"spread_10_2{suffix}" in df_range.columns:
    v_pp = last_val(f"spread_10_2{suffix}")
    d_spread = None if v_pp is None else v_pp * 100.0

def horizon_bp(col: str) -> float | None:
    c = f"{col}{suffix}"
    if c in df_range.columns:
        v = last_val(c)
        return None if v is None else v * 100.0
    return None

d_2y  = horizon_bp("y_2y")
d_10y = horizon_bp("y_10y")

# Huidig 10–2 niveau (pp)
if "spread_10_2" in df_range.columns:
    latest_spread = float(df_range.dropna(subset=["spread_10_2"]).iloc[-1]["spread_10_2"])
else:
    latest_spread = None

regime = "⏸️ Neutraal"; explanation = []
if (d_spread is not None) and (d_2y is not None) and (d_10y is not None):
    if d_spread >= thr_steepen:
        if d_2y <= 0 and d_10y >= 0:
            regime = "✅ Bull steepening"; explanation.append("Kort ↓ en Lang ↑ (easing/risico-on).")
        elif d_2y > 0 and d_10y > 0:
            regime = "⚠️ Bear steepening"; explanation.append("Beide ↑, lang harder (inflatiepremie).")
        else:
            regime = "ℹ️ Mixed steepening"
    elif d_spread <= -thr_steepen:
        if d_2y >= 0 and d_10y <= 0:
            regime = "❌ Bear flattening"; explanation.append("Kort ↑ en Lang ↓ (tightening/groei-stress).")
        elif d_2y < 0 and d_10y < 0:
            regime = "🟦 Bull flattening"; explanation.append("Beide ↓, lang harder (flight-to-quality).")
        else:
            regime = "ℹ️ Mixed flattening"

with sigL:
    label = f"{regime} — Δ{regime_horizon}(10Y–2Y): {round(d_spread or 0.0,1)} bp"
    if regime.startswith("✅"): st.success(label, icon="✅")
    elif regime.startswith("❌"): st.error(label, icon="❌")
    elif regime.startswith("⚠️"): st.warning(label, icon="⚠️")
    elif regime.startswith("🟦"): st.info(label, icon="ℹ️")
    else: st.info(label, icon="⏸️")
    if explanation: st.caption(" • ".join(explanation))
if latest_spread is not None and latest_spread < 0:
    st.warning(f"🔻 Inversie actief: 10Y–2Y = {round(latest_spread,2)} pp (negatief).", icon="🔻")
else:
    st.caption(f"10Y–2Y = { '—' if latest_spread is None else str(round(latest_spread,2)) + ' pp' }")

# ================== Spreads (synchroon met periode) ==================
st.subheader("Spreads")
if "spread_10_2" in df_range.columns or "spread_30_10" in df_range.columns:
    sp = go.Figure()
    if "spread_10_2" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_10_2"], name="10Y - 2Y"))
    if "spread_30_10" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_30_10"], name="30Y - 10Y"))
    sp.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
    sp.update_xaxes(range=[start_date, end_date])
    sp = add_recession_shapes(sp, pd.to_datetime(start_date), pd.to_datetime(end_date), show=show_recessions)
    st.plotly_chart(sp, use_container_width=True)
st.caption("10Y–2Y < 0 = inversie (vaak 6–18 mnd vóór recessies).")

# ================== Rentes + 1D Δ ==================
st.subheader("Rentes per looptijd (tijdreeks)")
avail_yields = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
default_sel  = [c for c in ["y_2y","y_10y","y_30y"] if c in avail_yields] or avail_yields[:2]
sel = st.multiselect("Selecteer looptijden", avail_yields, default=default_sel)

show_1d_delta = st.toggle(
    "Toon 1D delta (bp) onder de grafiek — één rij per looptijd",
    value=True,
    help="Gebruikt *_d1_bp indien aanwezig; anders diff()*100."
)

if sel:
    def get_1d_delta_bp(df_in: pd.DataFrame, col: str) -> pd.Series:
        dcol = f"{col}_d1_bp"
        if dcol in df_in.columns:
            return pd.to_numeric(df_in[dcol], errors="coerce")
        s = pd.to_numeric(df_in[col], errors="coerce")
        return s.diff() * 100.0

    n_delta_rows = len(sel) if show_1d_delta else 0
    total_rows   = 1 + n_delta_rows
    row_heights = [1.0] if n_delta_rows == 0 else [0.6] + [0.4 / n_delta_rows] * n_delta_rows

    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=0.06)

    for col in sel:
        fig.add_trace(go.Scatter(x=df_range["date"], y=df_range[col], name=col.upper(), mode="lines"),
                      row=1, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)

    if show_1d_delta:
        for i, col in enumerate(sel, start=2):
            dser = get_1d_delta_bp(df_range, col)
            colors = [("#16a34a" if (pd.notna(v) and v >= 0) else "#dc2626") for v in dser]
            fig.add_trace(go.Bar(x=df_range["date"], y=dser, name=f"{col.upper()} Δ1D (bp)",
                                 marker_color=colors, showlegend=False),
                          row=i, col=1)
            fig.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5, row=i, col=1)
            fig.update_yaxes(title_text=f"{col.upper()} Δ1D (bp)", row=i, col=1, zeroline=True)

    fig.update_xaxes(title_text="Date", row=total_rows, col=1, range=[start_date, end_date])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if show_recessions:
        fig = add_recession_shapes(fig, pd.to_datetime(start_date), pd.to_datetime(end_date), show=True)
    st.plotly_chart(fig, use_container_width=True)

# ================== Heatmap ==================
st.subheader("Heatmap van rentes")
avail_yields_hm = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
if avail_yields_hm:
    hm = df_range[["date"] + avail_yields_hm].set_index("date")
    hfig = go.Figure(data=go.Heatmap(
        z=hm[avail_yields_hm].T.values, x=hm.index.astype(str),
        y=[c.replace("y_","").upper() for c in avail_yields_hm], coloraxis="coloraxis"
    ))
    hfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_colorscale="Viridis")
    st.plotly_chart(hfig, use_container_width=True)

# ================== Deltas (7d/30d uit enriched view) ==================
st.header("Δ Deltas (basispunten)")
available_horizons = []
for name, suf in [("7d","_d7"), ("30d","_d30")]:
    if any([(f"{b}{suf}" in df_range.columns) for b in ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]]):
        available_horizons.append((name, suf))

# Delta-matrix (laatste dag)
if not df_range.empty and available_horizons:
    last_row = df_range.iloc[-1]
    base_items = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                  ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]
    labels, data_by_h = [], []
    for hname, suf in available_horizons:
        vals, labs = [], []
        for base, lab in base_items:
            col = f"{base}{suf}"
            if col in df_range.columns:
                v = last_row[col]
                vals.append(None if pd.isna(v) else float(v)*100.0)  # pp → bp
                labs.append(lab)
        if vals:
            data_by_h.append(vals); labels = labs
    if data_by_h:
        st.subheader("Delta-matrix (laatste dag)")
        hm2 = go.Figure(data=go.Heatmap(z=np.array(data_by_h).T, x=[h for h, _ in available_horizons],
                                        y=labels, coloraxis="coloraxis"))
        hm2.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                          coloraxis_colorscale="RdBu", coloraxis_cmid=0)
        st.plotly_chart(hm2, use_container_width=True)
        st.caption("Rood = stijging (bp), Blauw = daling. Waarden van de laatste dag.")

# Delta tijdreeks
st.subheader("Delta tijdreeks")
hoptions = [h for h, _ in available_horizons] or []
if hoptions:
    hsel = st.selectbox("Horizon", hoptions, index=0)
    suf = dict(available_horizons)[hsel]
    candidates, labels_map = [], {}
    for base, label in [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                        ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]:
        col = f"{base}{suf}"
        if col in df_range.columns:
            candidates.append(col); labels_map[col] = f"{label} ({hsel})"
    default_pick = [c for c in candidates if c.startswith("y_10y")] or candidates[:1]
    choose = st.multiselect("Kies metrics", candidates, default=default_pick,
                            format_func=lambda c: labels_map.get(c, c))
    if choose:
        figd = go.Figure()
        for c in choose:
            yvals_pp = pd.to_numeric(df_range[c], errors="coerce")
            yvals_bp = yvals_pp * 100.0
            colors = [("#16a34a" if (pd.notna(v) and v >= 0) else "#dc2626") for v in yvals_bp]
            figd.add_trace(go.Bar(x=df_range["date"], y=yvals_bp,
                                  name=labels_map.get(c, c), marker_color=colors,
                                  opacity=(0.8 if len(choose)>1 else 1)))
        figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
        figd.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                           barmode=("overlay" if len(choose)>1 else "group"),
                           yaxis_title="Δ (bp)", xaxis_title="Date")
        figd.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(figd, use_container_width=True)

# ================== Z-score 10Y–2Y + Histogram Δ1D ==================
st.subheader("Z-score 10Y–2Y en Δ1D histogram")
extrasL, extrasR = st.columns(2)

with extrasL:
    if "spread_10_2" in df_range.columns:
        s = pd.to_numeric(df_range["spread_10_2"], errors="coerce")
        roll = 252
        mu = s.rolling(roll, min_periods=60).mean()
        sd = s.rolling(roll, min_periods=60).std()
        z  = (s - mu) / sd
        figz = go.Figure()
        figz.add_trace(go.Scatter(x=df_range["date"], y=z, name="Z(10Y–2Y)"))
        figz.add_hline(y=0, line_dash="dot")
        figz.add_hrect(y0=-2, y1=2, fillcolor="LightGray", opacity=0.2, line_width=0)
        figz.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Z-score", xaxis_title="Date")
        figz.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(figz, use_container_width=True)

with extrasR:
    # Δ1D histogram (gebruik *_d1_bp indien aanwezig; anders diff()*100)
    cand = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
    if cand:
        tgt = st.selectbox("Δ1D histogram voor", cand, index=min(3, len(cand)-1))
        dcol = f"{tgt}_d1_bp"
        if dcol in df_range.columns:
            ser = pd.to_numeric(df_range[dcol], errors="coerce")
        else:
            ser = pd.to_numeric(df_range[tgt], errors="coerce").diff()*100
        ser = ser.replace([np.inf, -np.inf], np.nan).dropna()
        hfig = go.Figure(data=[go.Histogram(x=ser, nbinsx=40)])
        hfig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                           xaxis_title="Δ1D (bp)", yaxis_title="Aantal dagen")
        st.plotly_chart(hfig, use_container_width=True)

# ================== Tabel + download ==================
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_range.sort_values("date", ascending=False).round(round_dp))

csv = df_range.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd op periode)", data=csv,
                   file_name="yield_curve_filtered.csv", mime="text/csv")
