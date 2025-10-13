# pages/3_SPX_Options.py
# SPX Options — Skew/Delta/PPD met duidelijke "points-to-strike" x-as en robuuste berekeningen

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- BigQuery helpers ----------
try:
    from utils.bq import run_query, bq_ping  # jouw bestaande helper
except Exception:
    import google.cloud.bigquery as bq
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bq.Client(credentials=credentials, project=creds["project_id"])

    def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        job_config = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter(k, "STRING", str(v)) for k, v in (params or {}).items()
            ]
        )
        return client.query(sql, job_config=job_config).to_dataframe()

    def bq_ping() -> bool:
        try:
            _ = run_query("SELECT 1 AS ok")
            return True
        except Exception:
            return False

# ---------- Page config ----------
st.set_page_config(page_title="SPX Options", layout="wide")
st.title("🧰 SPX Options — Skew, Delta & PPD")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})
SPX_VIEW   = TABLES.get("spx_options_view", f"{PROJECT_ID}.marketdata.spx_options")  # override via secrets mogelijk

# ---------- Sidebar / Settings ----------
with st.sidebar:
    st.header("⚙️ Instellingen")

    st.markdown("**Strike-centrering**")
    center_mode = st.radio(
        "Kies referentie voor centrering",
        options=["Rounded (aanbevolen)", "ATM (live underlying)"],
        index=0,
        help=(
            "• **Rounded**: centreer rond een afgeronde strike (netter voor skew)\n"
            "• **ATM**: centreer rond de actuele underlying (kan ‘verschuiven’ door de dag heen)"
        ),
    )

    round_base = st.select_slider(
        "Rond strikes op",
        options=[25, 50, 100],
        value=25,
        help="Rond de strike van de underlying af op deze stapgrootte."
    )

    max_pts = st.slider(
        "Afstand tot (gecentreerde) strike (± punten)",
        min_value=50, max_value=1000, value=400, step=50,
        help="Filter strikes binnen ±X punten rondom de gekozen centers."
    )

    dte_pref = st.selectbox(
        "DTE-selectie voor skew",
        options=["Nearest", "0–7", "8–21", "22–45", "46–90", "90+"],
        index=0,
        help="Kies welke looptijd-bucket je in de skew wilt tonen."
    )

    r = st.number_input("Risicovrije rente r", value=0.00, step=0.25, help="Jaarbasis (decimaal).")
    q_input = st.number_input("Dividend/Index carry q", value=0.00, step=0.25, help="Jaarbasis (decimaal).")

    st.markdown("---")
    st.caption("Tip: zet **Rounded** + **±400–600** voor heldere skew-curves.\n"
               "Je kunt later fijnmaziger filteren op specifieke expiraties.")

with st.expander("ℹ️ Uitleg: ‘Points to strike’, ronding en skew"):
    st.markdown(
        """
**Points to strike (K − S₀)**  
We plotten de afstand tussen de **optiestrike (K)** en een **center** (S₀) in **punten**.  
- Negatief = strikes **onder** center (meestal put-zijde)  
- Positief = strikes **boven** center (meestal call-zijde)

**Waarom niet een ratio / moneyness?**  
Een ratio (bijv. K/S) is lastiger te lezen bij intraday moves. **Punten** blijven intuïtief: “±200 punten van de center”.

**Rounded vs ATM**  
- **Rounded** (standaard): S₀ is de **afgeronde** underlying (bijv. naar 25/50/100). Dit geeft stabiele skew-assen.  
- **ATM**: S₀ = actuele underlying → kan intraday verschuiven; nuttig voor zeer korte DTE.

**DTE-buckets**  
Voor skew kiezen we één bucket (Nearest of range) om ruis te beperken; term structure zie je in de tab ‘Term Structure’.
        """
    )

# ---------- Data load ----------
@st.cache_data(show_spinner=True, ttl=300)
def load_latest_snapshot() -> pd.DataFrame:
    sql = f"""
    WITH last_snap AS (
      SELECT MAX(snapshot_date) AS snapshot_date
      FROM `{SPX_VIEW}`
    )
    SELECT
      contractSymbol, type, expiration, days_to_exp,
      strike, underlying_price, lastPrice, bid, ask,
      impliedVolatility, openInterest, volume, vix, snapshot_date
    FROM `{SPX_VIEW}`
    WHERE snapshot_date = (SELECT snapshot_date FROM last_snap)
    """
    return run_query(sql)

@st.cache_data(show_spinner=True, ttl=600)
def load_history_for_ppd(days_back: int = 14) -> pd.DataFrame:
    sql = f"""
    SELECT
      contractSymbol, type, expiration, days_to_exp,
      strike, underlying_price, bid, ask, lastPrice,
      impliedVolatility, openInterest, volume, snapshot_date
    FROM `{SPX_VIEW}`
    WHERE snapshot_date >= DATE_SUB((SELECT MAX(snapshot_date) FROM `{SPX_VIEW}`), INTERVAL {days_back} DAY)
    """
    return run_query(sql)

df = load_latest_snapshot()
if df.empty:
    st.warning("Geen SPX-optiedata gevonden in de view/tabel. Controleer `tables.spx_options_view` in je secrets.")
    st.stop()

# ---------- Clean & derive ----------
def _to_float(s, clip_min=None, default=np.nan):
    x = pd.to_numeric(s, errors="coerce")
    if clip_min is not None:
        x = x.clip(lower=clip_min)
    return x.fillna(default)

df["bid"]  = _to_float(df["bid"],  clip_min=0.0, default=np.nan)
df["ask"]  = _to_float(df["ask"],  clip_min=0.0, default=np.nan)
df["last"] = _to_float(df["lastPrice"], clip_min=0.0, default=np.nan)
df["iv"]   = _to_float(df["impliedVolatility"], clip_min=0.0, default=np.nan)
df["strike"] = _to_float(df["strike"], clip_min=0.0, default=np.nan)
df["S"]      = _to_float(df["underlying_price"], clip_min=0.0, default=np.nan)
df["dte"]    = _to_float(df["days_to_exp"], clip_min=0.0, default=np.nan)
df["mid"]    = ((df["bid"] + df["ask"]) / 2.0).where((df["bid"].notna() & df["ask"].notna()), df["last"])
df["mid"]    = _to_float(df["mid"], clip_min=0.0, default=np.nan)

# IV sanity
df.loc[(~np.isfinite(df["iv"])) | (df["iv"] == 0), "iv"] = np.nan
df = df.dropna(subset=["S", "strike", "iv", "dte"]).copy()
if df.empty:
    st.error("Na opschonen blijft geen geldige rij over (S/strike/iv/dte). Controleer datakwaliteit.")
    st.stop()

# ---------- Centering ----------
S_now = float(np.nanmedian(df["S"]))
if center_mode.startswith("Rounded"):
    # Rond S_now naar de dichtstbijzijnde 'round_base'
    center = round_base * round(S_now / round_base)
else:
    center = S_now

df["pts_to_strike"] = df["strike"] - center  # K - S0

# Filter op ± max_pts
df = df.loc[df["pts_to_strike"].between(-max_pts, max_pts)].copy()
if df.empty:
    st.warning("Geen rijen binnen de ingestelde ± afstand. Vergroot ‘Afstand tot strike’.")
    st.stop()

# ---------- DTE bucket ----------
def in_bucket(dte: float, bucket: str) -> bool:
    if bucket == "Nearest":
        return True
    lo, hi = {
        "0–7":  (0, 7),
        "8–21": (8, 21),
        "22–45": (22, 45),
        "46–90": (46, 90),
        "90+":  (90, 10_000),
    }[bucket]
    return (dte >= lo) and (dte <= hi)

# Kies 1 DTE target voor skew (Nearest = dichtstbijzijnde DTE in de gefilterde set)
if dte_pref == "Nearest":
    target_dte = df["dte"].iloc[(df["dte"].values).argmin()]  # dichtstbijzijnde bij 0? — beter: mediane DTE
    # Correcter: kies de modale/dichtstbijzijnde DTE aan de onderkant
    target_dte = df.loc[df["dte"] == df["dte"].min(), "dte"].iloc[0]
    skew_df = df.loc[df["dte"] == target_dte].copy()
else:
    mask = df["dte"].apply(lambda x: in_bucket(x, dte_pref))
    skew_df = df.loc[mask].copy()

if skew_df.empty:
    st.warning("Geen rijen in de gekozen DTE-bucket. Kies een andere bucket of vergroot ± afstand.")
    st.stop()

# ---------- Greeks (vectorized delta) ----------
from scipy.stats import norm

def bs_delta_vectorized(S_arr, K_arr, IV_arr, T_arr, r_arr, q_arr, is_call_arr) -> np.ndarray:
    # safeguards
    S_arr = np.asarray(S_arr, dtype=float)
    K_arr = np.asarray(K_arr, dtype=float)
    IV_arr = np.asarray(IV_arr, dtype=float)
    T_arr = np.asarray(T_arr, dtype=float)
    is_call_arr = np.asarray(is_call_arr, dtype=bool)

    n = len(S_arr)
    if np.isscalar(r_arr): r_arr = np.full(n, r_arr, dtype=float)
    if np.isscalar(q_arr): q_arr = np.full(n, q_arr, dtype=float)
    r_arr = np.asarray(r_arr, dtype=float)
    q_arr = np.asarray(q_arr, dtype=float)

    eps = 1e-12
    sigma = np.maximum(IV_arr, eps)
    T = np.maximum(T_arr / 365.0, eps)  # T in jaren
    sqrtT = np.sqrt(T)

    d1 = (np.log(np.maximum(S_arr, eps) / np.maximum(K_arr, eps)) + (r_arr - q_arr + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    disc = np.exp(-q_arr * T)
    return np.where(is_call_arr, disc * norm.cdf(d1), -disc * norm.cdf(-d1)).astype(float)

skew_df = skew_df.reset_index(drop=True)
S_arr   = skew_df["S"].to_numpy()
K_arr   = skew_df["strike"].to_numpy()
IV_arr  = skew_df["iv"].to_numpy()
T_arr   = skew_df["dte"].to_numpy()
is_call = (skew_df["type"].str.lower() == "call").to_numpy()

deltas = bs_delta_vectorized(S_arr, K_arr, IV_arr, T_arr, r, q_input, is_call)
skew_df["delta"] = deltas

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Underlying (S₀)", f"{S_now:,.0f}")
col2.metric("Center", f"{center:,.0f}", help="Gebruikt voor K − S₀")
col3.metric("DTE voor skew", f"{skew_df['dte'].median():.0f} d")
col4.metric("Aantal optierijen", f"{len(skew_df):,}")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📈 Skew (IV & Δ vs punten)", "🧵 Term Structure", "🧮 PPD per DTE"])

# --- Tab 1: Skew ---
with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Implied Volatility (iv) vs Points to strike",
                                        "Delta vs Points to strike"))

    # IV skew
    for side, color_name in [("call", None), ("put", None)]:
        sub = skew_df.loc[skew_df["type"].str.lower() == side]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["pts_to_strike"],
                y=sub["iv"],
                mode="markers",
                name=f"IV {side}",
                hovertemplate=f"{side.upper()}<br>pts: %{{x:.0f}}<br>IV: %{{y:.2%}}<br>K: %{{customdata[0]:.0f}}<extra></extra>",
                customdata=np.c_[sub["strike"]],
            ),
            row=1, col=1
        )

    # Delta skew
    for side in ["call", "put"]:
        sub = skew_df.loc[skew_df["type"].str.lower() == side]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["pts_to_strike"],
                y=sub["delta"],
                mode="markers",
                name=f"Δ {side}",
                hovertemplate=f"{side.upper()}<br>pts: %{{x:.0f}}<br>Δ: %{{y:.2f}}<br>K: %{{customdata[0]:.0f}}<extra></extra>",
                customdata=np.c_[sub["strike"]],
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Points to strike (K − center)", row=2, col=1)
    fig.update_yaxes(title_text="IV", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Delta (±)", row=2, col=1)
    fig.update_layout(height=700, showlegend=True, margin=dict(t=60, b=40, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Tip: IV-skew laat vaak **smile/smirk** zien; delta-punten helpen bij het positioneren van strangles "
        "op een consistente afstand in punten i.p.v. vage ratios."
    )

# --- Tab 2: Term Structure (IV vs DTE, per moneyness band in punten) ---
with tab2:
    # Maak buckets in points-to-strike (bijv. OTM-put band, near-ATM, OTM-call)
    bins = [-10_000, -200, -50, 50, 200, 10_000]
    labels = ["Put far OTM (≤−200)", "Put near (−200..−50)", "Near ATM (−50..50)", "Call near (50..200)", "Call far OTM (≥200)"]
    df["pts_band"] = pd.cut(df["pts_to_strike"], bins=bins, labels=labels)

    ts = (
        df.groupby(["dte", "pts_band"], as_index=False)[["iv"]]
          .median()
          .dropna()
    )

    fig_ts = go.Figure()
    for band in labels:
        sub = ts.loc[ts["pts_band"] == band]
        if sub.empty: 
            continue
        fig_ts.add_trace(
            go.Scatter(
                x=sub["dte"], y=sub["iv"], mode="lines+markers", name=band,
                hovertemplate="DTE: %{x:.0f}d<br>IV: %{y:.1%}<extra></extra>"
            )
        )
    fig_ts.update_layout(
        height=500, margin=dict(t=40, b=40, l=40, r=20),
        xaxis_title="DTE (dagen)", yaxis_title="Median IV", yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption("Median IV per DTE per ‘points-band’. Hiermee zie je of IV vooral **kort** of **lang** in de curve beweegt.")

# --- Tab 3: PPD (premium per dag) ---
with tab3:
    hist_days = 10
    dfh = load_history_for_ppd(hist_days)
    if dfh.empty:
        st.info("Geen historische snapshots om PPD te tonen.")
    else:
        dfh["bid"]  = _to_float(dfh["bid"],  clip_min=0.0, default=np.nan)
        dfh["ask"]  = _to_float(dfh["ask"],  clip_min=0.0, default=np.nan)
        dfh["last"] = _to_float(dfh["lastPrice"], clip_min=0.0, default=np.nan)
        dfh["mid"]  = ((dfh["bid"] + dfh["ask"]) / 2.0).where((dfh["bid"].notna() & dfh["ask"].notna()), dfh["last"])
        dfh["mid"]  = _to_float(dfh["mid"], clip_min=0.0, default=np.nan)
        dfh["S"]    = _to_float(dfh["underlying_price"], clip_min=0.0, default=np.nan)
        dfh["dte"]  = _to_float(dfh["days_to_exp"], clip_min=0.0, default=np.nan)

        # zelfde center en filter toepassen zodat het vergelijkbaar is
        dfh["pts_to_strike"] = dfh["strike"] - center
        dfh = dfh[dfh["pts_to_strike"].between(-max_pts, max_pts)].copy()

        # PPD: mid / dte (alleen dte > 0)
        dfh = dfh[(dfh["mid"].notna()) & (dfh["dte"] > 0)].copy()
        dfh["ppd"] = dfh["mid"] / dfh["dte"]

        # Kies een compacte representatie: median PPD per snapshot_date en pts-band (near ATM)
        dfh["pts_band"] = pd.cut(dfh["pts_to_strike"], bins=[-50, 50], labels=["Near ATM"], include_lowest=True)
        ppd_ts = (
            dfh.loc[dfh["pts_band"].notna()]
               .groupby(["snapshot_date"], as_index=False)[["ppd"]]
               .median()
        )

        fig_ppd = go.Figure()
        fig_ppd.add_trace(
            go.Bar(
                x=ppd_ts["snapshot_date"], y=ppd_ts["ppd"],
                hovertemplate="Datum: %{x}<br>PPD: %{y:.2f}<extra></extra>",
                name="Median PPD (near ATM)"
            )
        )
        fig_ppd.update_layout(
            height=380, margin=dict(t=40, b=40, l=40, r=20),
            xaxis_title="Snapshot datum", yaxis_title="Premium per dag (≈ $/dag)"
        )
        st.plotly_chart(fig_ppd, use_container_width=True)
        st.caption("Snel beeld van **verdiencapaciteit per dag** rond ATM. Voor strangles kun je dit uitbreiden met meerdere bands.")

# ---------- Debug / Data peek ----------
with st.expander("🧪 Debug / data-peek"):
    st.write("Skew sample (eerste 10 rijen):")
    st.dataframe(skew_df.head(10))

    st.write("Distinct expirations in skew-set (top 10):")
    st.write(skew_df["expiration"].dropna().astype(str).value_counts().head(10))

    st.write("DTE stats (skew-set):")
    st.write(skew_df["dte"].describe())
