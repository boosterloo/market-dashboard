# pages/Yield_US.py — 🇺🇸 US-only Yield: Curve, Real, Breakeven (interactief & robuust)
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

# Optionele views (worden automatisch samengevoegd als aanwezig)
US_TIPS_VIEW = TABLES.get("us_tips_view", None)     # bevat real_10y / breakeven_10y / breakeven_5y
US_ACM_VIEW  = TABLES.get("us_acm_tp_view", None)   # bevat acm_term_premium_10y
US_FWD_VIEW  = TABLES.get("us_forward_view", None)  # bevat ntfs / 18m fwd 3m

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
# Data laden & mergen
# ─────────────────────────────────────────────────────────
with st.spinner("US data laden uit BigQuery…"):
    US = load_us_view(US_VIEW)
    TIPS = load_optional_view(US_TIPS_VIEW, ["real_10y","breakeven_10y","breakeven_5y"],
                              {"real_10y":"real_10y","breakeven_10y":"breakeven_10y","breakeven_5y":"breakeven_5y"})
    FWD  = load_optional_view(US_FWD_VIEW, ["ntfs","fwd_18m_3m_minus_3m","near_term_forward_spread"],
                              {"fwd_18m_3m_minus_3m":"ntfs","near_term_forward_spread":"ntfs"})
    ACM  = load_optional_view(US_ACM_VIEW, ["acm_term_premium_10y","acm_tp_10y"],
                              {"acm_tp_10y":"acm_term_premium_10y"})

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

preset = st.radio(
    "Presets",
    ["1W","1M","3M","6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"],
    horizontal=True, index=4
)

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
    date_range = st.slider("Selecteer periode (Custom)",
                           min_value=dmin.date(), max_value=dmax.date(),
                           value=(clamp(dmax - pd.DateOffset(years=1)).date(), dmax.date()))
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
sp10_2 = (y10 - y2) if y10 is not None and y2 is not None else None
ntfs = g("ntfs")
real10 = g("real_10y"); be10 = g("breakeven_10y")
acm = g("acm_term_premium_10y")

fmt     = lambda x, d=round_dp: "—" if x is None or np.isnan(x) else f"{round(float(x), d)}%"
fmt_pp  = lambda x: "—" if x is None or np.isnan(x) else f"{round(float(x), 2)} pp"

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("3M", fmt(y3m))
k2.metric("2Y", fmt(y2))
k3.metric("10Y", fmt(y10))
k4.metric("30Y", fmt(y30))
k5.metric("10Y–2Y", fmt_pp(sp10_2))
k6.metric("NTFS", fmt_pp(ntfs))
k7.metric("10Y Real", fmt(real10))
k8.metric("10Y Breakeven", fmt(be10))
if acm is not None and not np.isnan(acm):
    k9, = st.columns(1)
    k9.metric("ACM Term Premium (10Y)", fmt_pp(acm))

# ─────────────────────────────────────────────────────────
# Snapshot — Term structure (met vergelijking, robuust)
# ─────────────────────────────────────────────────────────
st.subheader("Term Structure — snapshot")

snap_dates = US["date"].tolist()
if not snap_dates:
    st.info("Geen datums beschikbaar in US dataset.")
    st.stop()

def nearest_index(dates, target_ts):
    if not dates: return 0
    diffs = [abs((pd.Timestamp(d) - pd.Timestamp(target_ts)).days) for d in dates]
    idx = int(np.argmin(diffs))  # cast naar Python int
    return max(0, min(idx, len(dates) - 1))

snap_primary_idx = int(len(snap_dates) - 1)
snap_primary = st.selectbox(
    "Peildatum", options=snap_dates, index=snap_primary_idx,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
)

compare_enabled_default = len(snap_dates) > 1
compare = st.checkbox("Vergelijk met 2e peildatum", value=compare_enabled_default,
                      disabled=not compare_enabled_default)

snap_secondary = None
if compare:
    tgt = pd.Timestamp(snap_primary) - pd.DateOffset(months=1)
    idx = nearest_index(snap_dates, tgt)
    snap_secondary = st.selectbox(
        "2e peildatum", options=snap_dates, index=int(idx),
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
# Tijdreeks — Levels (multiselect maturities)
# ─────────────────────────────────────────────────────────
st.subheader("Tijdreeks — Levels (selecteer termijnen)")

# beschikbare maturities (alleen kolommen die bestaan)
available_mats = [(c, n) for c, n in [
    ("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y")
] if c in US.columns]

# default: alle standaard curve-termijnen
default_sel = [n for c, n in available_mats]  # 3M/2Y/5Y/10Y/30Y indien aanwezig
label_to_col = {n: c for c, n in available_mats}

maturity_labels = [n for _, n in available_mats]
chosen_labels = st.multiselect(
    "Toon termijnen",
    options=maturity_labels,
    default=default_sel,
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
# Tijdreeks — Spreads & NTFS (met toggles)
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
# Tijdreeks — 10Y Nominaal vs Reëel & Breakeven (alleen als data bestaat)
# ─────────────────────────────────────────────────────────
if ("y_10y" in US.columns) or ("real_10y" in US.columns) or ("breakeven_10y" in US.columns):
    st.subheader("Tijdreeks — 10Y Nominaal vs Reëel & Breakeven")
    fig3 = go.Figure()
    if "y_10y" in US.columns:
        fig3.add_trace(go.Scatter(x=US["date"], y=US["y_10y"], name="10Y Nominaal", mode="lines"))
    if "real_10y" in US.columns:
        fig3.add_trace(go.Scatter(x=US["date"], y=US["real_10y"], name="10Y Reëel (TIPS)", mode="lines"))
    if "breakeven_10y" in US.columns:
        fig3.add_trace(go.Scatter(x=US["date"], y=US["breakeven_10y"], name="10Y Breakeven", mode="lines"))
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="%", xaxis_title="Date")
    fig3.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(fig3, use_container_width=True)

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

# Δ tijdreeks
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
