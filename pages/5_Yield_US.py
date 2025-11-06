# pages/Yield_US.py — 🇺🇸 US-only Yield: Curve, Real & Breakeven
# Robuuste versie: veilige defaults, geen ambigu "Bewijs: —", nette fallbacks
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# BigQuery
from google.cloud import bigquery
from google.oauth2 import service_account

# ── App ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🇺🇸 US Yield — Curve, Real & Breakeven", layout="wide")
st.title("🇺🇸 US Yield — Curve, Real & Breakeven")

SECRETS_SA = st.secrets.get("gcp_service_account", None)
TABLES     = st.secrets.get("tables", {})

PROJECT_ID = (SECRETS_SA or {}).get("project_id") or st.secrets.get("project_id") or ""
US_VIEW    = TABLES.get("us_yield_view", f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")
US_TIPS    = TABLES.get("us_tips_view", None)       # real_10y, breakeven_10y, breakeven_5y
US_ACM     = TABLES.get("us_acm_tp_view", None)     # acm_term_premium_10y
US_FWD     = TABLES.get("us_forward_view", None)    # ntfs / near_term_forward_spread

# ── Helpers BQ ───────────────────────────────────────────────────────────────
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
    return set(query_df(sql, {"tbl": tbl})["column_name"].tolist())

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
        st.warning("⚠️ Geen 2Y kolom gevonden (y_2y_synth/y_2y). Curve-analyse zal beperkt zijn.")
    sel = ["date"]
    for src, alias in [("y_3m","y_3m"), (y2 or "y_2y","y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
        if src in cols: sel.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
    for s in ["spread_10_2","spread_30_10"]:
        if s in cols: sel.append(f"SAFE_CAST({s} AS FLOAT64) AS {s}")
    for r in ["real_10y","breakeven_10y","breakeven_5y"]:
        if r in cols: sel.append(f"SAFE_CAST({r} AS FLOAT64) AS {r}")
    for f in ["ntfs","fwd_18m_3m_minus_3m","near_term_forward_spread"]:
        if f in cols: sel.append(f"SAFE_CAST({f} AS FLOAT64) AS ntfs")
    for a in ["acm_term_premium_10y","acm_tp_10y"]:
        if a in cols: sel.append(f"SAFE_CAST({a} AS FLOAT64) AS acm_term_premium_10y")
    bases = ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]
    for base in bases:
        if f"{base}_d1_bp" in cols: sel.append(f"SAFE_CAST({base}_d1_bp AS FLOAT64) AS {base}_d1_bp")
        if f"{base}_d7"    in cols: sel.append(f"SAFE_CAST({base}_d7    AS FLOAT64) AS {base}_d7")
        if f"{base}_d30"   in cols: sel.append(f"SAFE_CAST({base}_d30   AS FLOAT64) AS {base}_d30")
    if "snapshot_date" in cols: sel.append("snapshot_date")

    df = query_df(f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def load_optional_view(fqtn: str | None, cols_wanted: list[str], rename_map: dict[str,str]) -> pd.DataFrame:
    if not fqtn:
        return pd.DataFrame()
    cols = list_columns(fqtn)
    if not cols: return pd.DataFrame()
    sel = ["date"]
    for c in cols_wanted:
        if c in cols:
            sel.append(f"SAFE_CAST({c} AS FLOAT64) AS {rename_map.get(c,c)}")
    if len(sel) == 1: return pd.DataFrame()
    df = query_df(f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date")
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── Tijd helpers ─────────────────────────────────────────────────────────────
def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) is not None else ts

def nearest_on_or_before(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp:
    dates_sorted = sorted([_to_naive(pd.Timestamp(d)) for d in dates])
    t = _to_naive(pd.Timestamp(target))
    if not dates_sorted: return t
    if t <= dates_sorted[0]: return dates_sorted[0]
    for d in reversed(dates_sorted):
        if d <= t: return d
    return dates_sorted[-1]

def nearest_index(dates: list[pd.Timestamp], target: pd.Timestamp) -> int:
    target = _to_naive(pd.Timestamp(target))
    ds = [_to_naive(pd.Timestamp(d)) for d in dates]
    if not ds: return 0
    diffs = [abs((d - target).days) for d in ds]
    return int(np.argmin(diffs))

def normalize_utc_today() -> pd.Timestamp:
    now_utc = pd.Timestamp.utcnow()
    if getattr(now_utc, "tzinfo", None) is not None:
        now_utc = now_utc.tz_localize(None)
    return now_utc.normalize()

# ── Data ────────────────────────────────────────────────────────────────────
with st.spinner("US data laden uit BigQuery…"):
    US = load_us_view(US_VIEW)
    TIPS = load_optional_view(US_TIPS, ["real_10y","breakeven_10y","breakeven_5y"],
                              {"real_10y":"real_10y","breakeven_10y":"breakeven_10y","breakeven_5y":"breakeven_5y"})
    FWD  = load_optional_view(US_FWD,  ["ntfs","fwd_18m_3m_minus_3m","near_term_forward_spread"],
                              {"fwd_18m_3m_minus_3m":"ntfs","near_term_forward_spread":"ntfs"})
    ACM  = load_optional_view(US_ACM,  ["acm_term_premium_10y","acm_tp_10y"],
                              {"acm_tp_10y":"acm_term_premium_10y"})
    for extra in [TIPS, FWD, ACM]:
        if not extra.empty:
            US = pd.merge(US, extra, on="date", how="left")

if US.empty:
    st.error("Geen US data gevonden. Check je view/secrets.")
    st.stop()

# ── Controls ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.1, 1.1, 1])
with c1:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with c2:
    delta_h = st.radio("Δ-horizon", ["1d","7d","30d"], horizontal=True, index=1)
with c3:
    show_table = st.toggle("Tabel onderaan", value=False)

st.subheader("Periode")
dmin = max(pd.to_datetime("1990-01-01"), US["date"].min())
dmax = US["date"].max()

preset_options = ["1W","1M","3M","6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"]
preset = st.radio("Presets", preset_options, horizontal=True, index=preset_options.index("Custom"))

def clamp(ts): return max(dmin, ts)

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
    default_min = clamp(dmax - pd.DateOffset(months=3)).date()
    date_range = st.slider("Selecteer periode (Custom)", min_value=dmin.date(), max_value=dmax.date(),
                           value=(default_min, dmax.date()))
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

US = US[(US["date"]>=start_date) & (US["date"]<=end_date)].copy()
if US.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ── Kleine utilities voor 'laatste geldige' waardes ─────────────────────────
def last_valid_value(df: pd.DataFrame, col: str) -> tuple[pd.Timestamp|None, float|None]:
    if col not in df.columns: return None, None
    s = df[["date", col]].dropna().tail(1)
    if s.empty: return None, None
    return pd.Timestamp(s.iloc[0]["date"]), float(s.iloc[0][col])

def last_valid_pair(df: pd.DataFrame, colA: str, colB: str) -> tuple[pd.Timestamp|None, float|None, float|None]:
    if colA not in df.columns or colB not in df.columns:
        return None, None, None
    j = df[["date", colA, colB]].dropna()
    if j.empty: return None, None, None
    r = j.tail(1).iloc[0]
    return pd.Timestamp(r["date"]), float(r[colA]), float(r[colB])

def row_at_or_before(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    dt = nearest_on_or_before(df["date"].tolist(), ts)
    r = df[df["date"]==dt].tail(1)
    return None if r.empty else r.iloc[0]

# ── KPI’s ───────────────────────────────────────────────────────────────────
y3m_d, y3m = last_valid_value(US, "y_3m")
y2_d,  y2  = last_valid_value(US, "y_2y")
y10_d, y10 = last_valid_value(US, "y_10y")
y30_d, y30 = last_valid_value(US, "y_30y")

sp10_2 = None
if y10 is not None and y2 is not None:
    sp10_2 = y10 - y2
elif "spread_10_2" in US.columns and US["spread_10_2"].notna().any():
    _, sp10_2 = last_valid_value(US, "spread_10_2")

ntfs = last_valid_value(US, "ntfs")[1]
real10 = last_valid_value(US, "real_10y")[1]
be10 = last_valid_value(US, "breakeven_10y")[1]
acm = last_valid_value(US, "acm_term_premium_10y")[1]

fmt     = lambda x, d=round_dp: "—" if (x is None or pd.isna(x)) else f"{round(float(x), d)}%"
fmt_pp  = lambda x: "—" if (x is None or pd.isna(x)) else f"{round(float(x), 2)} pp"

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("3M", fmt(y3m))
k2.metric("2Y", fmt(y2))
k3.metric("10Y", fmt(y10))
k4.metric("30Y", fmt(y30))
k5.metric("10Y–2Y", fmt_pp(sp10_2))
k6.metric("NTFS", fmt_pp(ntfs))
k7.metric("10Y Reëel", fmt(real10))
k8.metric("10Y Breakeven", fmt(be10))
if acm is not None:
    k9, = st.columns(1)
    k9.metric("ACM Term Premium (10Y)", fmt_pp(acm))

# ── Snapshot — Term structure ────────────────────────────────────────────────
st.subheader("Term Structure — snapshot")
snap_dates = sorted(US["date"].tolist())
yesterday = normalize_utc_today() - pd.Timedelta(days=1)
default_primary = nearest_on_or_before(snap_dates, yesterday)
default_secondary = nearest_on_or_before(snap_dates, default_primary - pd.Timedelta(days=8))
snap_primary_idx = nearest_index(snap_dates, default_primary)
snap_secondary_idx = nearest_index(snap_dates, default_secondary)

snap_primary = st.selectbox("Peildatum", options=snap_dates, index=snap_primary_idx,
                            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
compare = st.checkbox("Vergelijk met 2e peildatum", value=True, disabled=(len(snap_dates)<=1))
snap_secondary = None
if compare:
    snap_secondary = st.selectbox("2e peildatum", options=snap_dates, index=snap_secondary_idx,
                                  format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))

def curve_points(row: pd.Series):
    mats = ["3M","2Y","5Y","10Y","30Y"]
    vals = [row.get("y_3m"), row.get("y_2y"), row.get("y_5y"), row.get("y_10y"), row.get("y_30y")]
    m = [m for m, v in zip(mats, vals) if pd.notna(v)]
    v = [v for v in vals if pd.notna(v)]
    return m, v

r1 = US[US["date"]==snap_primary].tail(1)
rowA = r1.iloc[0] if not r1.empty else pd.Series()
mA, vA = curve_points(rowA)

ts = make_subplots(rows=1, cols=2, subplot_titles=("Term structure", "Δ vs 2e peildatum (bp)"),
                   column_widths=[0.6, 0.4])

if mA:
    ts.add_trace(go.Scatter(x=mA, y=vA, mode="lines+markers", name=f"{pd.Timestamp(snap_primary).date()}"), row=1, col=1)

if compare and snap_secondary is not None:
    r2 = US[US["date"]==snap_secondary].tail(1)
    rowB = r2.iloc[0] if not r2.empty else pd.Series()
    mB, vB = curve_points(rowB)
    if mB:
        ts.add_trace(go.Scatter(x=mB, y=vB, mode="lines+markers",
                                name=f"{pd.Timestamp(snap_secondary).date()}",
                                line=dict(dash="dash")), row=1, col=1)
        # Δ-curve (bp)
        d, order = {}, {"3M":0,"2Y":1,"5Y":2,"10Y":3,"30Y":4}
        for m, val in zip(mA, vA): d[m] = [val, None]
        for m, val in zip(mB, vB): d[m] = [d.get(m,[None,None])[0], val]
        xs, ys = [], []
        for k in sorted(d.keys(), key=lambda x: order.get(x,99)):
            a, b = d[k]
            if a is not None and b is not None:
                xs.append(k); ys.append((a-b)*100.0)
        if xs:
            ts.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Δ (bp)"), row=1, col=2)

ts.update_yaxes(title_text="Yield (%)", row=1, col=1)
ts.update_yaxes(title_text="Δ (bp)", row=1, col=2)
ts.update_xaxes(title_text="Maturity", row=1, col=1)
ts.update_xaxes(title_text="Maturity", row=1, col=2)
ts.update_layout(margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(ts, use_container_width=True)

with st.expander("🔎 Uitleg: hoe lees je de curve en spreads?"):
    st.markdown(
        "- **Normaal (10Y > 2Y)**: groei & inflatieverwachting.\n"
        "- **Invers (10Y < 2Y)**: korte kant hoog (Fed), vaak *late-cycle*.\n"
        "- **Bear steepening**: lange einden stijgen sneller → inflatie/aanbod/term-premium.\n"
        "- **Bull steepening**: lange einden dalen sneller → recessie-/cut-verwachting.\n"
        "- **NTFS** is vaak *leading* en draait eerder dan 10Y–2Y.\n"
    )

# ── Curve-analyse (robuust, geen lege “Bewijs”) ─────────────────────────────
st.subheader("📊 Curve-analyse")

def spreads_from_row(row: pd.Series) -> tuple[float|None, float|None]:
    sp10_2 = row.get("spread_10_2")
    sp30_10 = row.get("spread_30_10")
    y2_, y10_, y30_ = row.get("y_2y"), row.get("y_10y"), row.get("y_30y")
    if (sp10_2 is None or pd.isna(sp10_2)) and pd.notna(y10_) and pd.notna(y2_):
        sp10_2 = float(y10_) - float(y2_)
    if (sp30_10 is None or pd.isna(sp30_10)) and pd.notna(y30_) and pd.notna(y10_):
        sp30_10 = float(y30_) - float(y10_)
    return (None if pd.isna(sp10_2) else float(sp10_2),
            None if pd.isna(sp30_10) else float(sp30_10))

last_row = US.dropna(subset=["y_10y"]).tail(1)
if not last_row.empty:
    last_row = last_row.iloc[0]
else:
    last_row = US.tail(1).iloc[0]

sp10_2_now, sp30_10_now = spreads_from_row(last_row)

# Referenties (1d/7d/30d terug, nearest on/before)
def ref_delta(days: int):
    rr = row_at_or_before(US, pd.Timestamp(last_row["date"]) - pd.Timedelta(days=days))
    if rr is None: return None, None
    p10_2, p30_10 = spreads_from_row(rr)
    d10_2 = None if (sp10_2_now is None or p10_2 is None) else sp10_2_now - p10_2
    d30_10 = None if (sp30_10_now is None or p30_10 is None) else sp30_10_now - p30_10
    return d10_2, d30_10

d1_10_2,  d1_30_10  = ref_delta(1)
d7_10_2,  d7_30_10  = ref_delta(7)
d30_10_2, d30_30_10 = ref_delta(30)

def fmt_pp(x): return "—" if (x is None or pd.isna(x)) else f"{round(float(x),2)} pp"
def fmt_bp(x): return "—" if (x is None or pd.isna(x)) else f"{round(float(x)*100,1)} bp"

# Logica:
# 1) Als 10Y & 2Y ontbreken → toon geen “vorm/regime” (leg uit wat mist).
# 2) Als 10Y–2Y bestaat, vul vorm + regime op basis van 7–30d Δ’s; anders val terug op 30Y–10Y.
lines = []

if sp10_2_now is None and (("y_2y" not in US.columns) or US["y_2y"].isna().all() or ("y_10y" not in US.columns) or US["y_10y"].isna().all()):
    st.info("ℹ️ 2Y en/of 10Y ontbreken in de huidige periode. Curve-analyse (10Y–2Y) wordt daarom niet getoond.")
else:
    # Vorm
    if sp10_2_now is None:
        shape_txt = "onbekend"
    elif sp10_2_now > 0:
        shape_txt = "normaal (10Y > 2Y)"
    elif sp10_2_now < -0.05:
        shape_txt = "duidelijk invers (10Y < 2Y)"
    else:
        shape_txt = "vlak of licht invers"

    # Regime: kies beste beschikbare horizon
    def pick_horizon():
        if d7_10_2 is not None and d7_30_10 is not None: return 7, d7_10_2, d7_30_10
        if d30_10_2 is not None and d30_30_10 is not None: return 30, d30_10_2, d30_30_10
        if d1_10_2 is not None and d1_30_10 is not None: return 1, d1_10_2, d1_30_10
        return None, None, None
    horizon, dA_10_2, dA_30_10 = pick_horizon()

    if horizon is None:
        st.markdown(f"**Vorm:** {shape_txt}  |  **10Y–2Y:** {fmt_pp(sp10_2_now)}  ·  **30Y–10Y:** {fmt_pp(sp30_10_now)}")
        st.caption("Niet genoeg waarnemingen voor een recente regime-inschatting.")
    else:
        if   dA_10_2 > 0 and dA_30_10 > 0: regime = "Bear steepening — lange einden lopen op"
        elif dA_10_2 < 0 and dA_30_10 < 0: regime = "Bull flattening — brede daling"
        elif dA_10_2 < 0 and dA_30_10 > 0: regime = "Bull steepening — lange rente daalt sneller"
        elif dA_10_2 > 0 and dA_30_10 < 0: regime = "Bear flattening — korte kant stijgt"
        else:                              regime = "Gemengd/geen dominant regime"

        st.markdown(
            f"**Vorm:** {shape_txt}  |  **10Y–2Y:** {fmt_pp(sp10_2_now)}  ·  **30Y–10Y:** {fmt_pp(sp30_10_now)}\n\n"
            f"**Regime (Δ{horizon}d):** {regime}  —  "
            f"Δ10Y–2Y: {fmt_bp(dA_10_2)},  Δ30Y–10Y: {fmt_bp(dA_30_10)}"
        )

# ── Tijdreeks — Levels (selecteer termijnen) ────────────────────────────────
st.subheader("Tijdreeks — Levels (selecteer termijnen)")
available_mats = [(c, n) for c, n in [
    ("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y")
] if c in US.columns]
default_sel = [n for c, n in available_mats]
label_to_col = {n: c for c, n in available_mats}
chosen_labels = st.multiselect("Toon termijnen", options=[n for _,n in available_mats], default=default_sel)
fig1 = go.Figure()
for lbl in chosen_labels:
    col = label_to_col.get(lbl)
    if col:
        fig1.add_trace(go.Scatter(x=US["date"], y=US[col], name=lbl, mode="lines"))
fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
fig1.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig1, use_container_width=True)

# ── Spreads & NTFS ──────────────────────────────────────────────────────────
st.subheader("Tijdreeks — 10Y–2Y, 30Y–10Y & NTFS")
scol1, scol2, scol3 = st.columns(3)
with scol1:
    show_10_2 = st.checkbox("Toon 10Y–2Y", value=("spread_10_2" in US.columns or (("y_10y" in US.columns) and ("y_2y" in US.columns))))
with scol2:
    show_30_10 = st.checkbox("Toon 30Y–10Y", value=("spread_30_10" in US.columns or (("y_30y" in US.columns) and ("y_10y" in US.columns))))
with scol3:
    show_ntfs = st.checkbox("Toon NTFS", value=("ntfs" in US.columns))
fig2 = go.Figure()
if show_10_2:
    if "spread_10_2" in US.columns and US["spread_10_2"].notna().any():
        fig2.add_trace(go.Scatter(x=US["date"], y=US["spread_10_2"], name="10Y–2Y", mode="lines"))
    elif {"y_10y","y_2y"}.issubset(US.columns):
        fig2.add_trace(go.Scatter(x=US["date"], y=US["y_10y"]-US["y_2y"], name="10Y–2Y (calc)", mode="lines"))
if show_30_10:
    if "spread_30_10" in US.columns and US["spread_30_10"].notna().any():
        fig2.add_trace(go.Scatter(x=US["date"], y=US["spread_30_10"], name="30Y–10Y", mode="lines"))
    elif {"y_30y","y_10y"}.issubset(US.columns):
        fig2.add_trace(go.Scatter(x=US["date"], y=US["y_30y"]-US["y_10y"], name="30Y–10Y (calc)", mode="lines"))
if show_ntfs and "ntfs" in US.columns:
    fig2.add_trace(go.Scatter(x=US["date"], y=US["ntfs"], name="NTFS", mode="lines"))
    fig2.add_hline(y=0.0, line_width=1, line_color="gray", opacity=0.5)
fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
fig2.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig2, use_container_width=True)

# ── 10Y Nominaal / Reëel / Breakeven ────────────────────────────────────────
has_nom10 = "y_10y" in US.columns and US["y_10y"].notna().any()
has_real  = "real_10y" in US.columns and US["real_10y"].notna().any()
has_be10  = "breakeven_10y" in US.columns and US["breakeven_10y"].notna().any()
if has_nom10 or has_real or has_be10:
    st.subheader("Tijdreeks — 10Y Nominaal" + (" vs Reëel & Breakeven" if (has_real or has_be10) else ""))
    fig3 = go.Figure()
    if has_nom10: fig3.add_trace(go.Scatter(x=US["date"], y=US["y_10y"], name="10Y Nominaal", mode="lines"))
    if has_real:  fig3.add_trace(go.Scatter(x=US["date"], y=US["real_10y"], name="10Y Reëel (TIPS)", mode="lines"))
    if has_be10:  fig3.add_trace(go.Scatter(x=US["date"], y=US["breakeven_10y"], name="10Y Breakeven", mode="lines"))
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="%", xaxis_title="Date")
    fig3.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("ℹ️ Reëel & breakeven — wat zie je hier?"):
        if has_real and has_be10:
            st.markdown("**Reëel** = TIPS-yield (10Y). **Breakeven** ≈ Nominaal − Reëel → impliciete inflatieverwachting. Let op: TIPS-liquiditeit/seasonality kan ruis geven.")
        elif not has_real and not has_be10:
            st.markdown("Alleen **nominale 10Y** beschikbaar. Voeg `real_10y` en/of `breakeven_10y` toe (FRED: DFII10 / T10YIE) voor volledige decompositie.")
        else:
            st.markdown("Grafiek toont de beschikbare componenten; compleet = zowel `real_10y` (TIPS) als `breakeven_10y`.")

# ── Deltas ──────────────────────────────────────────────────────────────────
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
if suf == "_d1_bp":
    dpp = USd / 100.0
    base = pd.to_numeric(US.get(b_sel), errors="coerce")
else:
    dpp  = pd.to_numeric(US.get(f"{b_sel}{suf}", pd.Series(index=US.index)), errors="coerce")
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

# ── Tabel & CSV ─────────────────────────────────────────────────────────────
if show_table:
    st.subheader("Tabel (US, gefilterd)")
    st.dataframe(US.sort_values("date", ascending=False).round(round_dp))

csv = US.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (US, gefilterd)", data=csv,
                   file_name="us_yield_filtered.csv", mime="text/csv")
