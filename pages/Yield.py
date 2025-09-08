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
# Tip: in secrets [tables].yield_view = "<project>.marketdata.yield_curve_dashboard_v"
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

# ---------- Compacte filterbalk ----------
# (Alles in één rij; extra opties in expander)
topA, topB, topC, topD = st.columns([1.2, 1, 1, 1.1])
with topA:
    st.caption("Filter")
    strict = st.toggle("Strikt (alle looptijden)", value=False,
                       help="Uit = alleen 2Y & 10Y verplicht.")
with topB:
    st.caption("Weergave")
    round_dp = st.slider("Decimalen", 1, 4, 2, label_visibility="collapsed")
with topC:
    st.caption("Tabel")
    show_table = st.toggle("Toon tabel", value=False)
with topD:
    with st.expander("Meer opties", expanded=False):
        st.markdown("- Recessies, deltas & extra grafieken stel je verderop in.")

# Basis filtering
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
left, right = st.columns([1.6, 1])
with left:
    preset = st.radio("Presets",
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

# ---------- Snapshot bij de Term Structure ----------
st.subheader("Term Structure")
snap_dates = list(df_range["date"].dropna().unique())
# compact select-slider; default = laatste datum
c1, c2 = st.columns([2, 1])
with c1:
    sel_date = st.select_slider("Snapshot datum", options=snap_dates, value=snap_dates[-1])
with c2:
    st.write("")  # spacer
    st.caption("Kies de datum voor de curve hieronder.")
snap = df_range[df_range["date"] == sel_date].tail(1)

# ---------- KPI’s boven de curve ----------
def fmt_pct(x):
    return "—" if pd.isna(x) else f"{round(float(x), round_dp)}%"

k1, k2, k3, k4, k5 = st.columns(5)
for col, box in zip(["y_3m","y_2y","y_5y","y_10y","y_30y"], [k1,k2,k3,k4,k5]):
    val = fmt_pct(snap[col].values[0]) if (col in snap.columns and not snap.empty) else "—"
    box.metric(col.upper().replace("_",""), val)

# ---------- Term Structure plot ----------
maturities, values = [], []
for col, label in [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y")]:
    if col in snap.columns:
        maturities.append(label)
        values.append(snap[col].values[0] if not snap.empty else None)
ts_fig = go.Figure()
ts_fig.add_trace(go.Scatter(x=maturities, y=values, mode="lines+markers"))
ts_fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Maturity")
st.plotly_chart(ts_fig, use_container_width=True)
st.caption("Normaal = stijgend (groei/inflatiepremie); vlak = late cyclus; invers = recessierisico/verwachte cuts.")

# ---------- Signals (steepening/flattening) ----------
st.subheader("Signals")
sigL, sigR = st.columns([1.3, 1])
with sigR:
    regime_horizon = st.radio("Regime-horizon", ["5d", "21d"], horizontal=True, index=1)
    suffix = "_delta_5d_bp" if regime_horizon == "5d" else "_delta_21d_bp"
    default_steepen = 8 if regime_horizon == "5d" else 10
    default_bigmove = 10 if regime_horizon == "5d" else 15
    st.caption(f"Drempels ({regime_horizon} Δ in basispunten)")
    thr_steepen = st.slider("Steil/plat drempel |Δ(10Y–2Y)|", 5, 50, default_steepen, step=1)
    thr_bigmove  = st.slider("Grote move per reeks", 5, 50, default_bigmove, step=1)

def last_val(col: str):
    if col not in df_range.columns: return None
    s = df_range[col].dropna()
    return None if s.empty else float(s.iloc[-1])

d_spread = None
spread_col = f"spread_10_2{suffix}"
if spread_col in df_range.columns:
    d_spread = last_val(spread_col)
elif {f"y_10y{suffix}", f"y_2y{suffix}"}.issubset(df_range.columns):
    d_spread = (last_val(f"y_10y{suffix}") or 0.0) - (last_val(f"y_2y{suffix}") or 0.0)

d_2y  = last_val(f"y_2y{suffix}")
d_10y = last_val(f"y_10y{suffix}")

if "spread_10_2" in df_f.columns:
    latest_spread = float(df_f.dropna(subset=["spread_10_2"]).iloc[-1]["spread_10_2"])
elif {"y_10y","y_2y"}.issubset(df_f.columns):
    tmp = df_f.dropna(subset=["y_10y","y_2y"])
    latest_spread = float(tmp.iloc[-1]["y_10y"] - tmp.iloc[-1]["y_2y"])
else:
    latest_spread = None

regime = "—"; explanation = []
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
    else:
        regime = "⏸️ Neutraal"

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

# ---------- Spreads (met recessies) ----------
st.subheader("Spreads")
if "spread_10_2" in df_range.columns or "spread_30_10" in df_range.columns:
    sp = go.Figure()
    if "spread_10_2" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_10_2"], name="10Y - 2Y"))
    if "spread_30_10" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_30_10"], name="30Y - 10Y"))
    sp.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
    # recessies
    if show_recessions:
        for start, end in NBER_RECESSIONS:
            s = pd.to_datetime(start); e = pd.to_datetime(end)
            sp.add_vrect(x0=s, x1=e, fillcolor="LightGray", opacity=0.25, layer="below", line_width=0)
    st.plotly_chart(sp, use_container_width=True)
st.caption("10Y–2Y < 0 = inversie (vaak 6–18 mnd vóór recessies).")

# ---------- Rentes per looptijd + 1D deltas ----------
st.subheader("Rentes per looptijd (tijdreeks)")
avail_yields = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
default_sel  = [c for c in ["y_2y","y_10y","y_30y"] if c in avail_yields] or avail_yields[:2]
sel = st.multiselect("Selecteer looptijden", avail_yields, default=default_sel)

show_1d_delta = st.toggle(
    "Toon 1D delta (bp) onder de grafiek — één rij per looptijd",
    value=True,
    help="Gebruikt *_delta_bp indien aanwezig, anders diff()*100."
)

if sel:
    def get_1d_delta_bp(df_in: pd.DataFrame, col: str) -> pd.Series:
        dcol = f"{col}_delta_bp"
        if dcol in df_in.columns:
            return pd.to_numeric(df_in[dcol], errors="coerce")
        s = pd.to_numeric(df_in[col], errors="coerce")
        return s.diff() * 100.0

    n_delta_rows = len(sel) if show_1d_delta else 0
    total_rows   = 1 + n_delta_rows
    row_heights = [1.0] if n_delta_rows == 0 else [0.55] + [0.45 / n_delta_rows] * n_delta_rows

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

    fig.update_xaxes(title_text="Date", row=total_rows, col=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if show_recessions:
        for start, end in NBER_RECESSIONS:
            s = pd.to_datetime(start); e = pd.to_datetime(end)
            fig.add_vrect(x0=s, x1=e, fillcolor="LightGray", opacity=0.25, layer="below", line_width=0,
                          row="all", col=1)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Heatmap ----------
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

# ================== DELTA-SECTIE ==================
st.header("Δ Deltas (basispunten)")
available_horizons = []
for hname, suffix_h in HORIZONS:
    if any([(f"{b}{suffix_h}" in df_range.columns) for _, b in DELTA_BASES]):
        available_horizons.append((hname, suffix_h))

# 4a) Delta-matrix heatmap (laatste dag)
try:
    last_row = df_range.iloc[-1]
    base_items = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                  ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]
    mat_labels, data_by_h = [], []
    for hname, suffix_h in available_horizons:
        vals, labs = [], []
        for base, label in base_items:
            col = f"{base}{suffix_h}"
            if col in df_range.columns:
                vals.append(last_row[col]); labs.append(label)
        if vals:
            data_by_h.append(vals); mat_labels = labs
    if data_by_h:
        st.subheader("Delta-matrix (laatste dag in gekozen periode)")
        hm2 = go.Figure(data=go.Heatmap(
            z=np.array(data_by_h).T, x=[h for h, _ in available_horizons],
            y=mat_labels, coloraxis="coloraxis"
        ))
        hm2.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                          coloraxis_colorscale="RdBu", coloraxis_cmid=0)
        st.plotly_chart(hm2, use_container_width=True)
        st.caption("Rood = stijging (bp), Blauw = daling. Waarden van de laatste dag.")
except Exception:
    pass

# 4b) Mini-bars per horizon (laatste dag)
if available_horizons:
    st.subheader("Delta-overzicht per horizon (laatste dag)")
    colL, colR = st.columns(2)
    target_cols = [colL, colR]
    base_items = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                  ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]
    for idx, (hname, suffix_h) in enumerate(available_horizons):
        vals, labs = [], []
        for base, label in base_items:
            col = f"{base}{suffix_h}"
            if col in df_range.columns:
                vals.append(last_row[col]); labs.append(label)
        if not vals: continue
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labs, y=vals, name=f"{hname} Δbp"))
        fig.update_layout(title=f"{hname} Δbp (laatste dag)",
                          margin=dict(l=10,r=10,t=40,b=10),
                          yaxis_title="Δ (bp)", xaxis_title="")
        target_cols[idx % 2].plotly_chart(fig, use_container_width=True)

# 4c) Delta tijdreeks — bars groen/rood + modus
st.subheader("Delta tijdreeks")
hoptions = [h for h, _ in available_horizons] or []
if hoptions:
    hsel = st.selectbox("Horizon", hoptions, index=0)
    suffix_sel = dict(available_horizons)[hsel]

    chart_mode = st.radio("Weergave", ["Overlay", "Gestapeld"], horizontal=True, index=0)

    candidates, labels_map = [], {}
    for base, label in [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                        ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]:
        col = f"{base}{suffix_sel}"
        if col in df_range.columns:
            candidates.append(col); labels_map[col] = f"{label} ({hsel})"

    default_pick = [c for c in candidates if c.startswith("y_10y")] or candidates[:1]
    choose = st.multiselect("Kies metrics", candidates, default=default_pick,
                            format_func=lambda c: labels_map.get(c, c))

    if choose:
        figd = go.Figure()
        for c in choose:
            yvals = pd.to_numeric(df_range[c], errors="coerce")
            colors = [("#16a34a" if (pd.notna(v) and v >= 0) else "#dc2626") for v in yvals]
            figd.add_trace(go.Bar(x=df_range["date"], y=yvals,
                                  name=labels_map.get(c, c),
                                  marker_color=colors,
                                  opacity=(0.75 if chart_mode == "Overlay" and len(choose) > 1 else 1.0)))
        figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
        figd.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                           barmode=("overlay" if chart_mode == "Overlay" else "relative"),
                           yaxis_title="Δ (bp)", xaxis_title="Date", legend_title_text="Reeks")
        st.plotly_chart(figd, use_container_width=True)

# ---------- EXTRA: Z-score 10Y–2Y + Histogram Δ1D ----------
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
        st.plotly_chart(figz, use_container_width=True)
        st.caption("Grijze band ≈ ±2σ over ~1 jaar handelsdagen.")

with extrasR:
    # kies looptijd voor histogram
    cand = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
    if cand:
        tgt = st.selectbox("Δ1D histogram voor", cand, index=min(3, len(cand)-1))
        ser = pd.to_numeric(df_range[tgt], errors="coerce").diff()*100
        hfig = go.Figure(data=[go.Histogram(x=ser.dropna(), nbinsx=40)])
        hfig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                           xaxis_title="Δ1D (bp)", yaxis_title="Aantal dagen")
        st.plotly_chart(hfig, use_container_width=True)

# ---------- Tabel + download ----------
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_range.sort_values("date", ascending=False).round(round_dp))

csv = df_range.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd op periode)", data=csv,
                   file_name="yield_curve_filtered.csv", mime="text/csv")
