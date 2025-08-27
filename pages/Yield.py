# pages/yield_curve.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.bq import run_query

st.set_page_config(page_title="Yield Curve Dashboard", layout="wide")
st.title("🧯 Yield Curve Dashboard")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})
# gebruik desgewenst je nieuwe view met delta-kolommen:
YIELD_VIEW = TABLES.get("yield_view", f"{PROJECT_ID}.marketdata.yield_curve_dashboard_v")

# ---------- Recessie-perioden (NBER, recente) ----------
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-30"),
]

# ---------- Kolommen ophalen & SELECT dynamisch opbouwen ----------
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

# yields
y2y_col = "y_2y_synth" if have("y_2y_synth") else ("y_2y" if have("y_2y") else None)
if not y2y_col:
    st.error(f"`{YIELD_VIEW}` bevat geen `y_2y_synth` of `y_2y`.")
    st.stop()

# basis select
select_parts = ["date"]
for src, alias in [("y_3m","y_3m"), (y2y_col,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
    if have(src): select_parts.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
if have("spread_10_2"):  select_parts.append("SAFE_CAST(spread_10_2 AS FLOAT64) AS spread_10_2")
if have("spread_30_10"): select_parts.append("SAFE_CAST(spread_30_10 AS FLOAT64) AS spread_30_10")
if have("snapshot_date"): select_parts.append("snapshot_date")

# delta-kolommen (yields + spreads)
DELTA_BASES = [
    ("y_3m","y_3m"), ("y_2y","y_2y"), ("y_5y","y_5y"),
    ("y_10y","y_10y"), ("y_30y","y_30y"),
    ("spread_10_2","spread_10_2"), ("spread_30_10","spread_30_10"),
]
HORIZONS = [("1d","_delta_bp"), ("5d","_delta_5d_bp"), ("21d","_delta_21d_bp")]

for src, alias in DELTA_BASES:
    for hname, suffix in HORIZONS:
        c = f"{src}{suffix}"
        if have(c):
            select_parts.append(f"SAFE_CAST({c} AS FLOAT64) AS {alias}{suffix}")

sql = f"SELECT {', '.join(select_parts)} FROM `{YIELD_VIEW}` ORDER BY date"

with st.spinner("Data ophalen uit BigQuery…"):
    df = run_query(sql, timeout=60)
if df.empty:
    st.warning("Geen data gevonden.")
    st.stop()

# ---------- Filters ----------
topA, topB, topC = st.columns([1.2,1,1])
with topA:
    strict = st.toggle("Strikt filter (alle looptijden aanwezig)", value=False,
                       help="Als uit, filtert alleen op aanwezigheid van 2Y & 10Y.")
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
    preset = st.radio(
        "Quick presets",
        ["Max", "YTD", "1Y", "3Y", "5Y", "10Y", "Custom"],
        horizontal=True,
        index=2,
    )
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
    date_range = st.slider(
        "Selecteer periode (Custom)",
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

# Extra KPI's
extra1, extra2 = st.columns(2)
# 10Y–2Y nu
if "spread_10_2" in df_f.columns:
    latest_spread = float(df_f.dropna(subset=["spread_10_2"]).iloc[-1]["spread_10_2"])
elif {"y_10y","y_2y"}.issubset(df_f.columns):
    tmp = df_f.dropna(subset=["y_10y","y_2y"]).copy()
    latest_spread = float(tmp.iloc[-1]["y_10y"] - tmp.iloc[-1]["y_2y"])
else:
    latest_spread = None
extra1.metric("10Y – 2Y (nu)", "—" if latest_spread is None else f"{round(latest_spread, 2)} pp",
              help="Positief = normale/steile curve; negatief = inversie (recessierisico).")

# Dagen inverted (12m)
inv_days = "—"
try:
    cutoff = dmax - pd.DateOffset(years=1)
    df_last12 = df_f[pd.to_datetime(df_f["date"]) >= cutoff].copy()
    if "spread_10_2" in df_last12.columns:
        inv_days = int((df_last12["spread_10_2"] < 0).sum())
    elif {"y_10y","y_2y"}.issubset(df_last12.columns):
        inv_days = int((df_last12["y_10y"] - df_last12["y_2y"] < 0).sum())
except Exception:
    pass
extra2.metric("Dagen inverted (12m)", inv_days,
              help="Aantal kalenderdagen in de laatste 12 maanden dat 10Y–2Y < 0.")

# ---------- Helper: Recessie-achtergrond ----------
def add_recession_shapes(fig: go.Figure, show: bool, x_start: pd.Timestamp, x_end: pd.Timestamp):
    if not show:
        return fig
    for start, end in NBER_RECESSIONS:
        s = pd.to_datetime(start); e = pd.to_datetime(end)
        if (e >= x_start) and (s <= x_end):
            fig.add_vrect(
                x0=max(s, x_start), x1=min(e, x_end),
                fillcolor="LightGray", opacity=0.25, layer="below", line_width=0,
            )
    return fig

# ---------- GRAFIEKEN (onder elkaar) ----------

# 1) Term Structure (snapshot)
st.subheader(f"Term Structure • {sel_date}")
maturities, values = [], []
order = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y")]
for col, label in order:
    if col in snap.columns:
        maturities.append(label)
        values.append(snap[col].values[0] if not snap.empty else None)
ts_fig = go.Figure()
ts_fig.add_trace(go.Scatter(x=maturities, y=values, mode="lines+markers", name="Snapshot"))
ts_fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Maturity")
st.plotly_chart(ts_fig, use_container_width=True)
st.markdown(
    """
**Wat je ziet:** de rentecurve (3M–30Y) op de gekozen datum.  
**Interpretatie:**
- *Normaal* → gezonde groei/verwachte inflatie.
- *Vlak* → einde rente-cyclus/ onzekerheid.
- *Invers* (kort > lang) → verhoogd recessierisico.
"""
)

# 2) Spreads (tijdreeks, met recessies)
st.subheader("Spreads")
if "spread_10_2" in df_range.columns or "spread_30_10" in df_range.columns:
    sp = go.Figure()
    if "spread_10_2" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_10_2"], name="10Y - 2Y"))
    if "spread_30_10" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_30_10"], name="30Y - 10Y"))
    sp.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
    sp = add_recession_shapes(sp, show_recessions, pd.to_datetime(start_date), pd.to_datetime(end_date))
    st.plotly_chart(sp, use_container_width=True)
else:
    st.info("Spreads niet beschikbaar in de view.")
st.markdown(
    """
**Interpretatie:** 10Y–2Y < 0 = inversie (vaak 6–18m vóór recessies). Herstel naar positief volgt vaak kort voor/tijdens krimp.
"""
)

# 3) Rentes per looptijd (tijdreeks, met recessies)
st.subheader("Rentes per looptijd (tijdreeks)")
avail_yields = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
default_sel = [c for c in ["y_2y","y_10y","y_30y"] if c in avail_yields] or avail_yields[:2]
sel = st.multiselect("Selecteer looptijden", avail_yields, default=default_sel)
if sel:
    yf = go.Figure()
    for col in sel:
        yf.add_trace(go.Scatter(x=df_range["date"], y=df_range[col], name=col.upper()))
    yf.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
    yf = add_recession_shapes(yf, show_recessions, pd.to_datetime(start_date), pd.to_datetime(end_date))
    st.plotly_chart(yf, use_container_width=True)

# 4) Heatmap (periode)
st.subheader("Heatmap van rentes")
hm = df_range[["date"] + avail_yields].set_index("date")
hfig = go.Figure(data=go.Heatmap(
    z=hm[avail_yields].T.values,
    x=hm.index.astype(str),
    y=[c.replace("y_","").upper() for c in avail_yields],
    coloraxis="coloraxis"
))
hfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_colorscale="Viridis")
st.plotly_chart(hfig, use_container_width=True)

# ================== DELTA-SECTIE ==================
st.header("Δ Deltas (basispunten)")

# Detecteer beschikbare delta-horizons
available_horizons = []
for hname, suffix in HORIZONS:
    # check minimaal één delta-kolom bestaat
    if any([(f"{b}{suffix}" in df_range.columns) for b_alias, b in DELTA_BASES]):
        available_horizons.append((hname, suffix))

# 4a) Delta-matrix heatmap (laatste dag in periode)
try:
    last_row = df_range.iloc[-1]
    mat_labels = []
    data_by_h = []
    base_items = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                  ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]
    for hname, suffix in available_horizons:
        vals = []
        labs = []
        for base, label in base_items:
            col = f"{base}{suffix}"
            if col in df_range.columns:
                vals.append(last_row[col])
                labs.append(label)
        if vals:
            data_by_h.append(vals)
            mat_labels = labs  # dezelfde volgorde aanhouden

    if data_by_h:
        st.subheader("Delta-matrix (laatste dag in gekozen periode)")
        hm2 = go.Figure(data=go.Heatmap(
            z=np.array(data_by_h).T,  # rijen = maturities/spreads, kolommen = horizons
            x=[h for h, _ in available_horizons],
            y=mat_labels,
            coloraxis="coloraxis"
        ))
        hm2.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_colorscale="RdBu", coloraxis_cmid=0)
        st.plotly_chart(hm2, use_container_width=True)
        st.caption("Rood = stijging (bp), Blauw = daling. Waarden van de laatste dag in je periode.")
except Exception:
    pass

# 4b) Mini-bars per horizon (laatste dag) – in 2 kolommen
if available_horizons:
    st.subheader("Delta-overzicht per horizon (laatste dag)")
    colL, colR = st.columns(2)
    tgt_cols = [colL, colR]
    base_items = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y"),
                  ("spread_10_2","10Y-2Y"), ("spread_30_10","30Y-10Y")]
    for idx, (hname, suffix) in enumerate(available_horizons):
        vals = []
        labs = []
        for base, label in base_items:
            col = f"{base}{suffix}"
            if col in df_range.columns:
                vals.append(last_row[col])
                labs.append(label)
        if not vals:
            continue
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labs, y=vals, name=f"{hname} Δbp"))
        fig.update_layout(
            title=f"{hname} Δbp (laatste dag)",
            margin=dict(l=10,r=10,t=40,b=10),
            yaxis_title="Δ (bp)", xaxis_title=""
        )
        tgt_cols[idx % 2].plotly_chart(fig, use_container_width=True)

# 4c) Delta tijdreeks
st.subheader("Delta tijdreeks")
# kies horizon
hoptions = [h for h, _ in available_horizons] or []
if hoptions:
    hsel = st.selectbox("Horizon", hoptions, index=0)
    suffix = dict(available_horizons)[hsel]
    # beschikbare metrics voor deze horizon
    candidates = []
    labels_map = {}
    for base, label in base_items:
        col = f"{base}{suffix}"
        if col in df_range.columns:
            candidates.append(col)
            labels_map[col] = f"{label} ({hsel})"
    default_pick = [c for c in candidates if c.startswith("y_10y")] or candidates[:1]
    choose = st.multiselect("Kies metrics", candidates, default=default_pick, format_func=lambda c: labels_map.get(c, c))
    if choose:
        figd = go.Figure()
        for c in choose:
            figd.add_trace(go.Scatter(x=df_range["date"], y=df_range[c], name=labels_map.get(c, c)))
        figd.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Δ (bp)", xaxis_title="Date")
        st.plotly_chart(figd, use_container_width=True)
        st.caption("Positief = stijging van de rente/spread over de gekozen horizon; negatief = daling.")
else:
    st.info("Geen delta-kolommen gevonden in de view (verwacht *_delta_bp, *_delta_5d_bp, *_delta_21d_bp).")

# ================== EINDE DELTA-SECTIE ==================

# Tabel + download (periode)
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_range.sort_values("date", ascending=False).round(round_dp))

csv = df_range.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd op periode)", data=csv,
                   file_name="yield_curve_filtered.csv", mime="text/csv")

with st.expander("Debug: kolommen in view"):
    st.write(sorted(list(cols)))
