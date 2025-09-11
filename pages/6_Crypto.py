# pages/4_Crypto.py
# Werkt met wide-view `marketdata.crypto_daily_wide` met kolommen:
# date, price_<asset>, delta_abs_<asset>, delta_pct_<asset>, ma7_<asset>, ma30_<asset>, ytd_pct_<asset>

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- BigQuery helpers ----
try:
    from utils.bq import run_query, bq_ping
except Exception:
    import google.cloud.bigquery as bq
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bq.Client(credentials=credentials, project=creds["project_id"])

    @st.cache_data(ttl=600, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

    def bq_ping() -> bool:
        try:
            _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
            return True
        except Exception:
            return False

# ---- Page config ----
st.set_page_config(page_title="🪙 Crypto Dashboard", layout="wide")
st.title("🪙 Crypto Dashboard")

# ---- Health check ----
if not bq_ping():
    st.error("Geen BigQuery-verbinding (controleer secrets/credentials).")
    st.stop()

# ---- Viewnaam ----
CRYPTO_WIDE_VIEW = st.secrets.get("tables", {}).get(
    "crypto_daily_wide",
    "nth-pier-468314-p7.marketdata.crypto_daily_wide"  # fallback
)

# ---- Data laden ----
@st.cache_data(ttl=600, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{CRYPTO_WIDE_VIEW}` ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # coerce numerics
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()
if df.empty:
    st.warning("Geen data in de view.")
    st.stop()

# ---- Assets & ordering ----
all_assets = sorted({c.split("_", 1)[1] for c in df.columns if c.startswith("price_")})
# Prioriteit (hoog → laag), rest alfabetisch
priority = ["btc", "eth", "sol", "bnb", "xrp", "ada", "dot", "avax", "link", "matic"]
ordered = [a for a in priority if a in all_assets] + [a for a in all_assets if a not in priority]

LABELS = {
    "btc": "BTC", "eth": "ETH", "sol": "SOL", "bnb": "BNB", "xrp": "XRP",
    "ada": "ADA", "dot": "DOT", "avax": "AVAX", "link": "LINK", "matic": "MATIC",
}
label_of = lambda a: LABELS.get(a, a.upper())

# ---- UI: periode (full width) + toggles ----
min_d, max_d = df["date"].min(), df["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)

with st.container():
    start, end = st.slider(
        "📅 Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )

collapse    = st.checkbox("Secties inklapbaar maken", value=False)
show_delta  = st.checkbox("Δ%-grafieken tonen", value=True)
show_combos = st.checkbox("Combinatiegrafieken tonen (BTC vs ETH, SOL vs BNB)", value=True)

# ---- Filter ----
mask = (df["date"] >= start) & (df["date"] <= end)
d = df.loc[mask].sort_values("date").copy()

# ---- Helpers ----
def col(asset: str, metric: str) -> str:
    return f"{metric}_{asset}"

def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

# ---- Kleuren (Okabe-Ito) ----
COLOR_PRICE   = "#111111"  # bijna zwart
COLOR_MA7     = "#E69F00"  # oranje
COLOR_MA30    = "#009E73"  # groen
COLOR_BAR_POS = "#009E73"  # groen
COLOR_BAR_NEG = "#D55E00"  # rood
COLOR_YTD     = "#CC79A7"  # paars

# ---- KPI's (laatste waarden top 4) ----
st.subheader("Kerncijfers")
top4 = ordered[:4] if len(ordered) >= 4 else ordered
kpi_cols = st.columns(len(top4) if top4 else 1)

for i, a in enumerate(top4):
    pcol = col(a, "price")
    dcol = col(a, "delta_pct")
    if pcol not in d.columns:
        continue
    sub = d[["date", pcol] + ([dcol] if dcol in d.columns else [])].dropna(subset=[pcol]).copy()
    if sub.empty:
        continue
    last_price = float(sub.iloc[-1][pcol]) if pd.notna(sub.iloc[-1][pcol]) else np.nan
    last_dpct = float(sub.iloc[-1][dcol]) if dcol in sub.columns and pd.notna(sub.iloc[-1][dcol]) else None

    with kpi_cols[i]:
        st.metric(
            label=label_of(a),
            value=f"{last_price:,.2f}" if pd.notna(last_price) else "—",
            delta=(f"{last_dpct:+.2f}%" if last_dpct is not None else "—")
        )

st.markdown("---")

# ---- Per asset ----
st.subheader("Per asset")
for a in ordered:
    price_c = col(a, "price")
    has_price = price_c in d.columns and d[price_c].notna().any()
    if not has_price:
        continue

    header = f"## {label_of(a)}"
    container = st.expander(header) if collapse else st.container()
    with container:
        if not collapse:
            st.markdown(header)

        # --- Prijs + MA7/MA30 ---
        ma7_c   = col(a, "ma7")
        ma30_c  = col(a, "ma30")

        subp = d[["date", price_c]].dropna().copy()
        subp[price_c] = _f(subp[price_c])

        # Als MA-kolommen bestaan, gebruik ze; anders reken ter plekke
        if ma7_c in d.columns:
            ma7 = _f(d[ma7_c]).reindex(d.index).loc[subp.index]
        else:
            ma7 = subp[price_c].rolling(7, min_periods=7).mean()

        if ma30_c in d.columns:
            ma30 = _f(d[ma30_c]).reindex(d.index).loc[subp.index]
        else:
            ma30 = subp[price_c].rolling(30, min_periods=30).mean()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=subp["date"], y=subp[price_c],
            name="Prijs", line=dict(width=2, color=COLOR_PRICE)
        ))
        if ma7.notna().any():
            fig1.add_trace(go.Scatter(
                x=subp["date"], y=ma7.values,
                name="MA7", line=dict(width=2, color=COLOR_MA7)
            ))
        if ma30.notna().any():
            fig1.add_trace(go.Scatter(
                x=subp["date"], y=ma30.values,
                name="MA30", line=dict(width=2, color=COLOR_MA30)
            ))
        fig1.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Δ% bars + YTD% lijn ---
        if show_delta:
            dcol  = col(a, "delta_pct")
            ytd_c = col(a, "ytd_pct")
            has_delta = dcol in d.columns and d[dcol].notna().any()
            has_ytd   = ytd_c in d.columns and d[ytd_c].notna().any()

            if has_delta or has_ytd:
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                if has_delta:
                    bars = d[["date", dcol]].dropna().copy()
                    bars[dcol] = _f(bars[dcol])
                    colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in bars[dcol].values]
                    fig2.add_trace(
                        go.Bar(x=bars["date"], y=bars[dcol], name="Δ% per dag",
                               marker=dict(color=colors), opacity=0.9),
                        secondary_y=False
                    )
                    try:
                        fig2.add_hline(y=0, line_dash="dot", opacity=0.6)
                    except Exception:
                        pass
                    fig2.update_yaxes(title_text="Δ% dag", secondary_y=False)

                if has_ytd:
                    ytd = d[["date", ytd_c]].dropna().copy()
                    ytd[ytd_c] = _f(ytd[ytd_c])
                    if has_delta:
                        # align op dezelfde X-as
                        ytd = ytd.set_index("date").reindex(pd.to_datetime(bars["date"])).reset_index().rename(columns={"index":"date"})
                    fig2.add_trace(
                        go.Scatter(x=ytd["date"], y=ytd[ytd_c], name="YTD%",
                                   line=dict(width=2, color=COLOR_YTD)),
                        secondary_y=True
                    )
                    fig2.update_yaxes(title_text="YTD%", secondary_y=True)

                fig2.update_layout(
                    height=320, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

# ---- Combo-grafieken (optioneel) ----
if show_combos:
    st.subheader("Combinatiegrafieken")

    def dual_axis(a_left: str, a_right: str, title: str, y_left: str | None = None, y_right: str | None = None):
        pL, pR = col(a_left, "price"), col(a_right, "price")
        if not all(c in d.columns for c in [pL, pR]):
            st.info(f"{title}: ontbrekende kolommen.")
            return
        sub = d[["date", pL, pR]].dropna(how="all").copy()
        if sub.empty:
            st.info(f"{title}: geen overlappende data.")
            return
        sub[pL] = _f(sub[pL]); sub[pR] = _f(sub[pR])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[pL], name=label_of(a_left),
                                 line=dict(width=2, color="#111111")), secondary_y=False)
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[pR], name=label_of(a_right),
                                 line=dict(width=2, color="#0072B2")), secondary_y=True)
        if y_left:  fig.update_yaxes(title_text=y_left,  secondary_y=False)
        if y_right: fig.update_yaxes(title_text=y_right, secondary_y=True)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                          title=title,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

    # BTC vs ETH
    if all(c in all_assets for c in ["btc", "eth"]):
        dual_axis("btc", "eth", "BTC vs ETH", y_left="BTC (USD)", y_right="ETH (USD)")

    # SOL vs BNB
    if all(c in all_assets for c in ["sol", "bnb"]):
        dual_axis("sol", "bnb", "SOL vs BNB", y_left="SOL (USD)", y_right="BNB (USD)")

# ---- Tabel (laatste 200 rijen binnen selectie, voor alle assets) ----
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date"]
for a in ordered:
    show_cols += [c for c in [
        col(a, "price"), col(a, "ma7"), col(a, "ma30"),
        col(a, "delta_pct"), col(a, "ytd_pct"), col(a, "delta_abs")
    ] if c in d.columns]
show_cols = [c for c in show_cols if c in d.columns]
st.dataframe(d[show_cols].tail(200), use_container_width=True)
