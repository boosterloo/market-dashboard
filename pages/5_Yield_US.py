# pages/Yield_US.py — 🇺🇸 US-only Yield: Curve, Real & Breakeven
# (interactief, robust, default=Custom(3M), snapshots=Yesterday & ~1w back)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# BigQuery (direct, zonder utils)
from google.cloud import bigquery
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="🇺🇸 US Yield — Curve, Real & Breakeven", layout="wide")
st.title("🇺🇸 US Yield — Curve, Real & Breakeven")

SECRETS_SA = st.secrets.get("gcp_service_account", None)
TABLES     = st.secrets.get("tables", {})

PROJECT_ID = (SECRETS_SA or {}).get("project_id") or st.secrets.get("project_id") or ""
US_VIEW = TABLES.get("us_yield_view", f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")

# Optioneel: extra views (automatisch gemerged wanneer aanwezig)
US_TIPS_VIEW = TABLES.get("us_tips_view", None)     # real_10y / breakeven_10y / breakeven_5y
US_ACM_VIEW  = TABLES.get("us_acm_tp_view", None)   # acm_term_premium_10y
US_FWD_VIEW  = TABLES.get("us_forward_view", None)  # ntfs / 18m fwd 3m

# ─────────────────────────────────────────────────────────
# BQ client & helpers
# ─────────────────────────────────────────────────────────
def make_bq_client():
    if SECRETS_SA:
        creds = service_account.Credentials.from_service_account_info(SECRETS_SA)
        return bigquery.Client(project=PROJECT_ID, credentials=creds)
    return bigquery.Client(project=PROJECT_ID or None)

CLIENT = make_bq_client()

@st.cache_data(ttl=1800, show_spinner=False)
def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig()
    if params:
        cfg.query_parameters = [bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params.items()]
    return CLIENT.query(sql, job_config=cfg).to_dataframe()

@st.cache_data(ttl=1800, show_spinner=False)
def list_columns(fqtn: str) -> set[str]:
    try:
        proj, dset, tbl = fqtn.split(".")
    except ValueError:
        return set()
    sql = f"""
    SELECT LOWER(column_name) AS column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = query_df(sql, {"tbl": tbl})
    return set(dfc["column_name"].tolist())

def pick_2y(cols: set[str]) -> str | None:
    return "y_2y_synth" if "y_2y_synth" in cols else ("y_2y" if "y_2y" in cols else None)

@st.cache_data(ttl=1800, show_spinner=True)
def load_us_view(fqtn: str) -> pd.DataFrame:
    cols = list_columns(fqtn)
    if not cols:
        st.error(f"View niet gevonden of geen kolommen: `{fqtn}`")
        return pd.DataFrame()

    y2 = pick_2y(cols)
    if not y2:
        st.error(f"`{fqtn}` mist 2Y kolom (y_2y_synth of y_2y).")
        return pd.DataFrame()

    sel = ["date"]
    # Niveaus
    for src, alias in [("y_3m","y_3m"), (y2,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
        if src in cols: sel.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
    # Spreads
    for s in ["spread_10_2","spread_30_10"]:
        if s in cols: sel.append(f"SAFE_CAST({s} AS FLOAT64) AS {s}")
    # Reëel & Breakeven (indien in dezelfde view)
    for r in ["real_10y", "breakeven_10y", "breakeven_5y"]:
        if r in cols: sel.append(f"SAFE_CAST({r} AS FLOAT64) AS {r}")
    # NTFS (aliasen naar ntfs)
    for f in ["ntfs", "fwd_18m_3m_minus_3m", "near_term_forward_spread"]:
        if f in cols: sel.append(f"SAFE_CAST({f} AS FLOAT64) AS ntfs")
    # ACM term premium
    for a in ["acm_term_premium_10y", "acm_tp_10y"]:
        if a in cols: sel.append(f"SAFE_CAST({a} AS FLOAT64) AS acm_term_premium_10y")
    # Deltas (bp/pp)
    bases = ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]
    for base in bases:
        if f"{base}_d1_bp" in cols: sel.append(f"SAFE_CAST({base}_d1_bp AS FLOAT64) AS {base}_d1_bp")
        if f"{base}_d7"    in cols: sel.append(f"SAFE_CAST({base}_d7    AS FLOAT64) AS {base}_d7")
        if f"{base}_d30"   in cols: sel.append(f"SAFE_CAST({base}_d30   AS FLOAT64) AS {base}_d30")
    if "snapshot_date" in cols: sel.append("snapshot_date")

    sql = f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date"
    df = query_df(sql)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def load_optional_view(fqtn: str | None, cols_wanted: list[str], rename_map: dict[str,str]) -> pd.DataFrame:
    if not fqtn:
        return pd.DataFrame()
    cols = list_columns(fqtn)
    if not cols:
        return pd.DataFrame()
    sel = ["date"]
    for c in cols_wanted:
        if c in cols:
            alias = rename_map.get(c, c)
            sel.append(f"SAFE_CAST({c} AS FLOAT64) AS {alias}")
    if len(sel) == 1:
        return pd.DataFrame()
    sql = f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date"
    df = query_df(sql)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ─────────────────────────────────────────────────────────
# TZ / selectbox helpers (voorkomt invalid index type errors)
# ─────────────────────────────────────────────────────────
def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) is not None else ts

def nearest_on_or_before(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp:
    dates_sorted = sorted([_to_naive(pd.Timestamp(d)) for d in dates])
    t = _to_naive(pd.Timestamp(target))
    if not dates_sorted:
        return t
    if t <= dates_sorted[0]:
        return dates_sorted[0]
    for d in reversed(dates_sorted):
        if d <= t:
            return d
    return dates_sorted[-1]

def nearest_index(dates: list[pd.Timestamp], target: pd.Timestamp) -> int:
    # robuust: kies index met minimale absolute dagafstand
    target = _to_naive(pd.Timestamp(target))
    ds = [_to_naive(pd.Timestamp(d)) for d in dates]
    if not ds:
        return 0
    diffs = [abs((d - target).days) for d in ds]
    return int(np.argmin(diffs))

def normalize_utc_today() -> pd.Timestamp:
    now_utc = pd.Timestamp.utcnow()
    if getattr(now_utc, "tzinfo", None) is not None:
        now_utc = now_utc.tz_localize(None)
    return now_utc.normalize()

# ─────────────────────────────────────────────────────────
# Data laden & mergen
# ─────────────────────────────────────────────────────────
with st.spinner("US data laden uit BigQuery…"):
    US = load_us_view(US_VIEW)
    TIPS = load_optional_view(
        US_TIPS_VIEW,
        ["real_10y","breakeven_10y","breakeven_5y"],
        {"real_10y":"real_10y","breakeven_10y":"breakeven_10y","breakeven_5y":"breakeven_5y"}
    )
    FWD  = load_optional_view(
        US_FWD_VIEW,
        ["ntfs","fwd_18m_3m_minus_3m","near_term_forward_spread"],
        {"fwd_18m_3m_minus_3m":"ntfs","near_term_forward_spread":"ntfs"}
    )
    ACM  = load_optional_view(
        US_ACM_VIEW,
        ["acm_term_premium_10y","acm_tp_10y"],
        {"acm_tp_10y":"acm_term_premium_10y"}
    )
    for extra in [TIPS, FWD, ACM]:
        if not extra.empty:
            US = pd.merge(US, extra, on="date", how="left")

if US.empty:
    st.error("Geen US data gevonden. Check je view/secrets.")
    st.stop()

# ─────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.1, 1.1, 1])
with c1:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with c2:
    delta_h = st.radio("Δ-horizon", ["1d","7d","30d"], horizontal=True, index=1,
                       help="Kies de horizon voor delta-plots (bp / %).")
with c3:
    show_table = st.toggle("Tabel onderaan", value=False)

st.subheader("Periode")
dmin = max(pd.to_datetime("1990-01-01"), US["date"].min())
dmax = US["date"].max()

# preset standaard op Custom (laatste 3 maanden)
preset_options = ["1W","1M","3M","6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"]
preset_default_index = preset_options.index("Custom")
preset = st.radio("Presets", preset_options, horizontal=True, index=preset_default_index)

def clamp(ts): 
    return max(dmin, ts)

if preset == "1W":   start_date, end_date = clamp(dmax - pd.DateOffset(weeks=1)), dmax
elif preset == "1M": start_date, end_date = clamp(dmax - pd.DateOffset(months=1)), dmax
elif preset == "3M": start_date, end_date = clamp(dmax - pd.DateOffset(months=3)), dmax
elif preset == "6M": start_date, end_date = clamp(dmax - pd.DateOffset(months=6)), dmax
elif preset == "1Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=1)), dmax
elif preset == "3Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=3)), dmax
elif preset == "5Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=5)), dmax
elif preset == "10Y":start_date, end_date = clamp(dmax - pd.DateOffset(years=10)), dmax
elif preset == "YTD":start_date, end_date = clamp(pd.Timestamp(dmax.year,1,1)), dmax
elif preset == "Max":start_date, end_date = dmin, dmax
else:
    # Default voor Custom: laatste 3 maanden
    default_min = clamp(dmax - pd.DateOffset(months=3)).date()
    date_range = st.slider(
        "Selecteer periode (Custom)",
        min_value=dmin.date(), max_value=dmax.date(),
        value=(default_min, dmax.date())
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

US = US[(US["date"]>=start_date) & (US["date"]<=end_date)].copy()
if US.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ─────────────────────────────────────────────────────────
# KPI’s
# ─────────────────────────────────────────────────────────
latest = US.iloc[-1]
def g(col):
    if col not in US.columns: return None
    val = latest.get(col)
    return None if pd.isna(val) else float(val)

y3m = g("y_3m"); y2 = g("y_2y"); y5 = g("y_5y"); y10 = g("y_10y"); y30 = g("y_30y")
sp10_2_chip = (y10 - y2) if y10 is not None and y2 is not None else None
ntfs_chip = g("ntfs")
real10_chip = g("real_10y"); be10_chip = g("breakeven_10y")
acm_chip = g("acm_term_premium_10y")

fmt     = lambda x, d=round_dp: "—" if x is None or np.isnan(x) else f"{round(float(x), d)}%"
fmt_pp  = lambda x: "—" if x is None or np.isnan(x) else f"{round(float(x), 2)} pp"

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("3M", fmt(y3m))
k2.metric("2Y", fmt(y2))
k3.metric("10Y", fmt(y10))
k4.metric("30Y", fmt(y30))
k5.metric("10Y–2Y", fmt_pp(sp10_2_chip))
k6.metric("NTFS", fmt_pp(ntfs_chip))
k7.metric("10Y Reëel", fmt(real10_chip))
k8.metric("10Y Breakeven", fmt(be10_chip))
if acm_chip is not None and not np.isnan(acm_chip):
    k9, = st.columns(1)
    k9.metric("ACM Term Premium (10Y)", fmt_pp(acm_chip))

# ─────────────────────────────────────────────────────────
# Regime-badge (compact + kleur) direct onder KPI’s
# ─────────────────────────────────────────────────────────
def _mk_spreads_from_levels(row: pd.Series):
    sp10_2 = row.get("spread_10_2")
    sp30_10 = row.get("spread_30_10")
    y2_, y10_, y30_ = row.get("y_2y"), row.get("y_10y"), row.get("y_30y")
    if (sp10_2 is None or pd.isna(sp10_2)) and pd.notna(y10_) and pd.notna(y2_):
        sp10_2 = float(y10_) - float(y2_)
    if (sp30_10 is None or pd.isna(sp30_10)) and pd.notna(y30_) and pd.notna(y10_):
        sp30_10 = float(y30_) - float(y10_)
    sp10_2 = None if (sp10_2 is None or pd.isna(sp10_2)) else float(sp10_2)
    sp30_10 = None if (sp30_10 is None or pd.isna(sp30_10)) else float(sp30_10)
    return sp10_2, sp30_10

_last = US.iloc[-1]
sp10_2_now, sp30_10_now = _mk_spreads_from_levels(_last)

def _nearest_row(days_back: int):
    tgt = pd.Timestamp(_last["date"]) - pd.Timedelta(days=days_back)
    dt = nearest_on_or_before(list(US["date"]), tgt)
    r = US[US["date"] == dt].tail(1)
    return None if r.empty else r.iloc[0]

row7  = _nearest_row(7)  or _nearest_row(30) or _nearest_row(1)
if row7 is not None:
    sp10_2_prev, sp30_10_prev = _mk_spreads_from_levels(row7)
else:
    sp10_2_prev, sp30_10_prev = sp10_2_now, sp30_10_now

d10_2 = None if (sp10_2_now is None or sp10_2_prev is None) else sp10_2_now - sp10_2_prev
d30_10 = None if (sp30_10_now is None or sp30_10_prev is None) else sp30_10_now - sp30_10_prev

if sp10_2_now is None:
    shape_txt, shape_color = "onbekend", "#6b7280"  # gray
elif sp10_2_now > 0:
    shape_txt, shape_color = "normaal", "#10b981"   # green
elif sp10_2_now < -0.05:
    shape_txt, shape_color = "invers", "#ef4444"    # red
else:
    shape_txt, shape_color = "vlak", "#f59e0b"      # amber

def _significant(x, thr_bp): 
    return (x is not None) and (abs(x*100) >= thr_bp)
thr_bp = 10  # ≈7–10d drempel

if   (d10_2 is None or d30_10 is None):
    regime_txt, regime_color = "onduidelijk", "#6b7280"
elif (d10_2 > 0 and d30_10 > 0 and (_significant(d10_2,thr_bp) or _significant(d30_10,thr_bp))):
    regime_txt, regime_color = "bear steepening", "#ef4444"
elif (d10_2 < 0 and d30_10 < 0 and (_significant(d10_2,thr_bp) or _significant(d30_10,thr_bp))):
    regime_txt, regime_color = "bull flattening", "#10b981"
elif (d10_2 < 0 and d30_10 > 0 and (_significant(d10_2,thr_bp) or _significant(d30_10,thr_bp))):
    regime_txt, regime_color = "bull steepening", "#10b981"
elif (d10_2 > 0 and d30_10 < 0 and (_significant(d10_2,thr_bp) or _significant(d30_10,thr_bp))):
    regime_txt, regime_color = "bear flattening", "#ef4444"
else:
    regime_txt, regime_color = "gemengd", "#6b7280"

badge_html = f"""
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:4px;">
  <span style="background:{shape_color};color:white;padding:4px 8px;border-radius:999px;font-weight:600;">
    Regime: {shape_txt}
  </span>
  <span style="background:{regime_color};color:white;padding:4px 8px;border-radius:999px;">
    {regime_txt}
  </span>
  <span style="color:#6b7280;">Δ7–10d 10Y–2Y: {('—' if d10_2 is None else str(round(d10_2*100,1))+' bp')}, 30Y–10Y: {('—' if d30_10 is None else str(round(d30_10*100,1))+' bp')}</span>
</div>
"""
st.markdown(badge_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Snapshot — Term structure (met vergelijking, robuust)
# ─────────────────────────────────────────────────────────
st.subheader("Term Structure — snapshot")
snap_dates = sorted(US["date"].tolist())
if not snap_dates:
    st.info("Geen datums beschikbaar in US dataset.")
    st.stop()

# defaults: gisteren en ~een week terug (8 dagen), gebruik robuuste indexbepaling
yesterday = normalize_utc_today() - pd.Timedelta(days=1)
default_primary = nearest_on_or_before(snap_dates, yesterday)
default_secondary = nearest_on_or_before(snap_dates, default_primary - pd.Timedelta(days=8))
snap_primary_idx = nearest_index(snap_dates, default_primary)
snap_secondary_idx = nearest_index(snap_dates, default_secondary)

snap_primary = st.selectbox(
    "Peildatum", options=snap_dates, index=snap_primary_idx,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
)

compare = st.checkbox("Vergelijk met 2e peildatum", value=True, disabled=(len(snap_dates) <= 1))
snap_secondary = None
if compare:
    snap_secondary = st.selectbox(
        "2e peildatum", options=snap_dates, index=snap_secondary_idx,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
    )

def curve_points(row: pd.Series):
    mats = ["3M","2Y","5Y","10Y","30Y"]
    vals = [row.get("y_3m"), row.get("y_2y"), row.get("y_5y"), row.get("y_10y"), row.get("y_30y")]
    m = [m for m, v in zip(mats, vals) if pd.notna(v)]
    v = [v for v in vals if pd.notna(v)]
    return m, v

r1 = US[US["date"] == snap_primary].tail(1)
rowA = r1.iloc[0] if not r1.empty else pd.Series()
mA, vA = curve_points(rowA)

ts = make_subplots(rows=1, cols=2, subplot_titles=("Term structure", "Δ vs 2e peildatum (bp)"),
                   column_widths=[0.6, 0.4])

if mA:
    ts.add_trace(go.Scatter(x=mA, y=vA, mode="lines+markers",
                            name=f"{pd.Timestamp(snap_primary).date()}"), row=1, col=1)

if compare and snap_secondary is not None:
    r2 = US[US["date"] == snap_secondary].tail(1)
    rowB = r2.iloc[0] if not r2.empty else pd.Series()
    mB, vB = curve_points(rowB)
    if mB:
        ts.add_trace(go.Scatter(x=mB, y=vB, mode="lines+markers",
                                name=f"{pd.Timestamp(snap_secondary).date()}",
                                line=dict(dash="dash")), row=1, col=1)
        # Δ-curve (bp)
        d = {}
        for m, val in zip(mA, vA): d[m] = [val, None]
        for m, val in zip(mB, vB): d[m] = [d.get(m, [None, None])[0], val]
        order = {k: i for i, k in enumerate(["3M", "2Y", "5Y", "10Y", "30Y"])}
        xs, ys = [], []
        for k in sorted(d.keys(), key=lambda x: order.get(x, 99)):
            a, b = d[k]
            if a is not None and b is not None:
                xs.append(k); ys.append((a - b) * 100.0)
        if xs:
            ts.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Δ (bp)"), row=1, col=2)

ts.update_yaxes(title_text="Yield (%)", row=1, col=1)
ts.update_yaxes(title_text="Δ (bp)", row=1, col=2)
ts.update_xaxes(title_text="Maturity", row=1, col=1)
ts.update_xaxes(title_text="Maturity", row=1, col=2)
ts.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(ts, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Uitleg + Sterke, data-gedreven curve-analyse
# ─────────────────────────────────────────────────────────
st.markdown(
    "### 🔎 Hoe lees je de curve?\n"
    "- **Normaal (10Y > 2Y)**: groei & inflatieverwachting in de lange kant.\n"
    "- **Invers (10Y < 2Y)**: korte kant hoog (Fed), vaak *late-cycle*.\n"
    "- **Bear steepening**: lange einden **stijgen** sneller → inflatie/aanbod/term-premium.\n"
    "- **Bull steepening**: lange einden **dalen** sneller → recessie-/cut-verwachting.\n"
    "- **NTFS** draait vaak vroeg; nuttig als *leading* signaal.\n"
)

def _pct_rank(series: pd.Series, value: float) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or value is None or pd.isna(value):
        return None
    return float((s <= value).mean())

def _fmt(x, dp=2, suffix="pp"):
    return "—" if x is None or pd.isna(x) else f"{round(float(x), dp)} {suffix}"

def _nearest(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp:
    return nearest_on_or_before(dates, target)

def _get_spreads_row(row: pd.Series) -> tuple[float|None,float|None]:
    sp10_2 = row.get("spread_10_2")
    sp30_10 = row.get("spread_30_10")
    y2_, y10_, y30_ = row.get("y_2y"), row.get("y_10y"), row.get("y_30y")
    if (sp10_2 is None or pd.isna(sp10_2)) and pd.notna(y10_) and pd.notna(y2_):
        sp10_2 = float(y10_) - float(y2_)
    if (sp30_10 is None or pd.isna(sp30_10)) and pd.notna(y30_) and pd.notna(y10_):
        sp30_10 = float(y30_) - float(y10_)
    sp10_2 = None if (sp10_2 is None or pd.isna(sp10_2)) else float(sp10_2)
    sp30_10 = None if (sp30_10 is None or pd.isna(sp30_10)) else float(sp30_10)
    return sp10_2, sp30_10

# huidige punt + referentiepunten
last_row = US.iloc[-1]
dates_list = list(US["date"])
y10_now = float(last_row["y_10y"]) if "y_10y" in US.columns and pd.notna(last_row.get("y_10y")) else None
sp10_2_now, sp30_10_now = _get_spreads_row(last_row)

def row_at_delta(days: int) -> pd.Series | None:
    tgt = pd.Timestamp(last_row["date"]) - pd.Timedelta(days=days)
    dt = _nearest(dates_list, tgt)
    r = US[US["date"] == dt].tail(1)
    return None if r.empty else r.iloc[0]

row_1d  = row_at_delta(1)
row_7d  = row_at_delta(7)
row_30d = row_at_delta(30)

def spread_deltas(prev_row: pd.Series | None):
    if prev_row is None:
        return None, None
    sp10_2_prev, sp30_10_prev = _get_spreads_row(prev_row)
    d10_2 = None if (sp10_2_now is None or sp10_2_prev is None) else (sp10_2_now - sp10_2_prev)
    d30_10 = None if (sp30_10_now is None or sp30_10_prev is None) else (sp30_10_now - sp30_10_prev)
    return d10_2, d30_10

d1_10_2,  d1_30_10  = spread_deltas(row_1d)
d7_10_2,  d7_30_10  = spread_deltas(row_7d)
d30_10_2, d30_30_10 = spread_deltas(row_30d)

# thresholds (significantie)
THR_1D_BP   = 5     # ≥5bp in 1d
THR_7D_BP   = 10    # ≥10bp in 7d
THR_30D_BP  = 20    # ≥20bp in 30d
def _is_sig(d_now: float|None, thr_bp: float) -> bool:
    if d_now is None or pd.isna(d_now):
        return False
    return abs(d_now*100.0) >= thr_bp

# percentielen over laatste 3 jaar (context)
cutoff_3y = pd.Timestamp(last_row["date"]) - pd.DateOffset(years=3)
win = US[US["date"] >= cutoff_3y]
p_spread = _pct_rank(win["spread_10_2"] if "spread_10_2" in win.columns else pd.Series(dtype=float), sp10_2_now) if sp10_2_now is not None else None
p_10y    = _pct_rank(win["y_10y"]        if "y_10y"        in win.columns else pd.Series(dtype=float), y10_now)    if y10_now    is not None else None

# NTFS signaal (optioneel)
ntfs_now = float(last_row["ntfs"]) if "ntfs" in US.columns and pd.notna(last_row.get("ntfs")) else None
ntfs_flag = None
if ntfs_now is not None:
    if ntfs_now > 0.0:
        ntfs_flag = "NTFS > 0 (minder cuts geprijsd op korte horizon)"
    elif ntfs_now < 0.0:
        ntfs_flag = "NTFS < 0 (cuts geprijsd in de nabije toekomst)"

# Reëel/breakeven driver-hint (optioneel)
real10 = float(last_row["real_10y"]) if "real_10y" in US.columns and pd.notna(last_row.get("real_10y")) else None
be10   = float(last_row["breakeven_10y"]) if "breakeven_10y" in US.columns and pd.notna(last_row.get("breakeven_10y")) else None

def driver_hint() -> str | None:
    if y10_now is None or (real10 is None and be10 is None):
        return None
    # Vergelijk met ~7d terug
    r7_real = float(row_7d.get("real_10y")) if (row_7d is not None and "real_10y" in row_7d.index and pd.notna(row_7d.get("real_10y"))) else None
    r7_be   = float(row_7d.get("breakeven_10y")) if (row_7d is not None and "breakeven_10y" in row_7d.index and pd.notna(row_7d.get("breakeven_10y"))) else None
    parts = []
    if real10 is not None and r7_real is not None:
        parts.append("reëel ↑" if real10 > r7_real else "reëel ↓")
    if be10 is not None and r7_be is not None:
        parts.append("breakeven ↑" if be10 > r7_be else "breakeven ↓")
    if not parts:
        return None
    if ("reëel ↑" in parts and "breakeven ↑" in parts):
        return "Nominale 10Y gedreven door **zowel** reële rente als inflatieverwachting."
    if ("reëel ↑" in parts and "breakeven ↓" in parts):
        return "Nominale 10Y vooral **reëel** gedreven (groei/term premium)."
    if ("reëel ↓" in parts and "breakeven ↑" in parts):
        return "Nominale 10Y vooral **inflatieverwachting** (breakeven) gedreven."
    return " / ".join(parts)

# Vorm en regime (tekst + bewijs)
if sp10_2_now is None:
    shape_txt = "onbekend (ontbrekende 2Y/10Y)"
elif sp10_2_now > 0:
    shape_txt = "normaal (10Y > 2Y)"
elif sp10_2_now < -0.05:
    shape_txt = "duidelijk invers (10Y < 2Y)"
else:
    shape_txt = "vlak of licht invers"

def describe_trend():
    # Kies 7d als hoofd-horizon, val terug op 30d of 1d
    dA_10_2, dA_30_10, thr = d7_10_2, d7_30_10, THR_7D_BP
    if dA_10_2 is None or dA_30_10 is None:
        dA_10_2, dA_30_10, thr = d30_10_2, d30_30_10, THR_30D_BP
    if dA_10_2 is None or dA_30_10 is None:
        dA_10_2, dA_30_10, thr = d1_10_2, d1_30_10, THR_1D_BP

    if dA_10_2 is None or dA_30_10 is None:
        return "onduidelijk (onvoldoende data)", "—"

    sig10_2 = _is_sig(dA_10_2, thr)
    sig30_10 = _is_sig(dA_30_10, thr)

    if dA_10_2 > 0 and dA_30_10 > 0 and (sig10_2 or sig30_10):
        regime = "Bear steepening — lange einden lopen op"
    elif dA_10_2 < 0 and dA_30_10 < 0 and (sig10_2 or sig30_10):
        regime = "Bull flattening — brede daling in spreads"
    elif dA_10_2 < 0 and dA_30_10 > 0 and (sig10_2 or sig30_10):
        regime = "Bull steepening — lange rente daalt sneller"
    elif dA_10_2 > 0 and dA_30_10 < 0 and (sig10_2 or sig30_10):
        regime = "Bear flattening — korte kant stijgt"
    else:
        regime = "Gemengd/geen dominant regime"
    evidence = f"Δ7d 10Y–2Y {_fmt(d7_10_2)} · 30Y–10Y {_fmt(d7_30_10)} | Δ30d 10Y–2Y {_fmt(d30_10_2)} · 30Y–10Y {_fmt(d30_30_10)}"
    return regime, evidence

regime_txt2, evidence_txt = describe_trend()

# Contextregels
ctx_lines = []
if p_spread is not None:
    ctx_lines.append(f"10Y–2Y in **p{int(round(p_spread*100))}** over 3Y (lager = meer invers).")
if p_10y is not None:
    ctx_lines.append(f"10Y nominaal in **p{int(round(p_10y*100))}** over 3Y.")
if ntfs_flag:
    ctx_lines.append(ntfs_flag)
drv = driver_hint()
if drv:
    ctx_lines.append(drv)

st.markdown(
    "### 📊 Curve-analyse\n"
    f"- **Vorm:** {shape_txt}  |  **10Y–2Y:** {_fmt(sp10_2_now)}  ·  **30Y–10Y:** {_fmt(sp30_10_now)}\n"
    f"- **Regime:** {regime_txt2}\n"
    f"- **Bewijs:** {evidence_txt}\n"
    + ("- " + "\n- ".join(ctx_lines) if ctx_lines else "")
)

# ─────────────────────────────────────────────────────────
# Tijdreeks — Levels (selecteer termijnen)
# ─────────────────────────────────────────────────────────
st.subheader("Tijdreeks — Levels (selecteer termijnen)")
available_mats = [(c, n) for c, n in [
    ("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y")
] if c in US.columns]
default_sel = [n for c, n in available_mats]
label_to_col = {n: c for c, n in available_mats}
maturity_labels = [n for _, n in available_mats]
chosen_labels = st.multiselect(
    "Toon termijnen", options=maturity_labels, default=default_sel,
    help="Standaard staan de curve-termijnen aan. Vink uit om te vereenvoudigen."
)
fig1 = go.Figure()
for lbl in chosen_labels:
    col = label_to_col.get(lbl)
    if col:
        fig1.add_trace(go.Scatter(x=US["date"], y=US[col], name=lbl, mode="lines"))
fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
fig1.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig1, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Spreads & NTFS
# ─────────────────────────────────────────────────────────
st.subheader("Tijdreeks — 10Y–2Y, 30Y–10Y & NTFS")
scol1, scol2, scol3 = st.columns(3)
with scol1:
    show_10_2 = st.checkbox("Toon 10Y–2Y", value=("spread_10_2" in US.columns))
with scol2:
    show_30_10 = st.checkbox("Toon 30Y–10Y", value=("spread_30_10" in US.columns))
with scol3:
    show_ntfs = st.checkbox("Toon NTFS", value=("ntfs" in US.columns))
fig2 = go.Figure()
if show_10_2 and "spread_10_2" in US.columns:
    fig2.add_trace(go.Scatter(x=US["date"], y=US["spread_10_2"], name="10Y–2Y", mode="lines"))
if show_30_10 and "spread_30_10" in US.columns:
    fig2.add_trace(go.Scatter(x=US["date"], y=US["spread_30_10"], name="30Y–10Y", mode="lines"))
if show_ntfs and "ntfs" in US.columns:
    fig2.add_trace(go.Scatter(x=US["date"], y=US["ntfs"], name="NTFS", mode="lines"))
    fig2.add_hline(y=0.0, line_width=1, line_color="gray", opacity=0.5)
fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
fig2.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# 10Y Nominaal vs Reëel & Breakeven — toon alleen wat echt bestaat
# ─────────────────────────────────────────────────────────
has_real = "real_10y" in US.columns and US["real_10y"].notna().any()
has_be10 = "breakeven_10y" in US.columns and US["breakeven_10y"].notna().any()
has_nom10 = "y_10y" in US.columns and US["y_10y"].notna().any()

if has_nom10 or has_real or has_be10:
    title = "Tijdreeks — 10Y Nominaal" + (" vs Reëel & Breakeven" if (has_real or has_be10) else "")
    st.subheader(title)
    fig3 = go.Figure()
    if has_nom10:
        fig3.add_trace(go.Scatter(x=US["date"], y=US["y_10y"], name="10Y Nominaal", mode="lines"))
    if has_real:
        fig3.add_trace(go.Scatter(x=US["date"], y=US["real_10y"], name="10Y Reëel (TIPS)", mode="lines"))
    if has_be10:
        fig3.add_trace(go.Scatter(x=US["date"], y=US["breakeven_10y"], name="10Y Breakeven", mode="lines"))
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="%", xaxis_title="Date")
    fig3.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("ℹ️ Reëel & breakeven — wat zie je hier?"):
        if has_real and has_be10:
            st.markdown("**Reëel** = TIPS-yield (10Y). **Breakeven** ≈ Nominaal − Reëel → impliciete inflatieverwachting. Let op liquiditeit/seasonality in TIPS; kijk vooral naar trend/niveaus.")
        elif not has_real and not has_be10:
            st.markdown("Alleen **nominale 10Y** beschikbaar. Voeg `real_10y` en/of `breakeven_10y` toe (bijv. FRED: DFII10 & T10YIE) om meer te zien.")
        else:
            st.markdown("De grafiek toont wat beschikbaar is; compleet = zowel `real_10y` (TIPS) als `breakeven_10y`.")

# ─────────────────────────────────────────────────────────
# Deltas — histogram & tijdreeks
# ─────────────────────────────────────────────────────────
st.subheader("Deltas — histogram & tijdreeks")

if   delta_h == "1d": suf="_d1_bp"
elif delta_h == "7d": suf="_d7"
else:                  suf="_d30"

bases = [("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y"),
         ("spread_10_2","10Y-2Y"),("spread_30_10","30Y-10Y")]
def_idx = next((i for i,(b,_) in enumerate(bases) if b=="y_10y"), 0)
b_sel, label_sel = st.selectbox("Metric", bases, index=def_idx, format_func=lambda t: t[1])

def get_delta_series(df: pd.DataFrame, base: str) -> pd.Series:
    if suf == "_d1_bp":
        if f"{base}_d1_bp" in df.columns:
            return pd.to_numeric(df[f"{base}_d1_bp"], errors="coerce")
        return pd.to_numeric(df.get(base), errors="coerce").diff() * 100.0
    else:
        col = f"{base}{suf}"
        if col not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce") * 100.0  # pp → bp

USd = get_delta_series(US, b_sel)

# Relatief (%): Δpp / vorige pp * 100
if suf == "_d1_bp":
    dpp = USd / 100.0
    base = pd.to_numeric(US.get(b_sel), errors="coerce")
else:
    dpp = pd.to_numeric(US.get(f"{b_sel}{suf}", pd.Series(index=US.index)), errors="coerce")
    base = pd.to_numeric(US.get(b_sel), errors="coerce")
pct = (dpp / base.shift(1).replace(0,np.nan)) * 100.0

h1, h2 = st.columns(2)
with h1:
    H = go.Figure()
    H.add_trace(go.Histogram(x=USd.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="US", opacity=0.7))
    H.update_layout(title=f"Δ {label_sel} — absoluut (bp)", barmode="overlay",
                    margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Δ (bp)", yaxis_title="Aantal dagen")
    st.plotly_chart(H, use_container_width=True)
with h2:
    H2 = go.Figure()
    H2.add_trace(go.Histogram(x=pct.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="US", opacity=0.7))
    H2.update_layout(title=f"Δ {label_sel} — relatief (%)",
                     margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Δ (%)", yaxis_title="Aantal dagen")
    st.plotly_chart(H2, use_container_width=True)

figd = go.Figure()
figd.add_trace(go.Bar(x=US["date"], y=USd, name=f"US Δ{delta_h}", opacity=0.7))
figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
figd.update_layout(margin=dict(l=10,r=10,t=10,b=10), barmode="overlay", yaxis_title="Δ (bp)", xaxis_title="Date")
figd.update_xaxes(range=[start_date, end_date])
st.plotly_chart(figd, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Tabel & download
# ─────────────────────────────────────────────────────────
if show_table:
    st.subheader("Tabel (US, gefilterd)")
    st.dataframe(US.sort_values("date", ascending=False).round(round_dp))

csv = US.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (US, gefilterd)", data=csv,
                   file_name="us_yield_filtered.csv", mime="text/csv")
