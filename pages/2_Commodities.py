# ==============================
# Combinatiegrafieken (Energy & Metals)
# Plak dit ergens NA je df/kleuren/helpers (bv. na "Per instrument")
# ==============================

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.markdown("---")
st.subheader("Combinatiegrafieken")

# Kleuren (Okabe-Ito) – los van je MA-kleuren
COLOR_WTI    = "#111111"  # bijna zwart
COLOR_BRENT  = "#0072B2"  # blauw
COLOR_GAS    = "#009E73"  # groen
COLOR_GOLD   = "#E69F00"  # oranje
COLOR_SILVER = "#56B4E9"  # sky blue (of kies grijs: "#999999")

def _to_float(s):
    return pd.to_numeric(s, errors="coerce").astype(float)

# ---------- Energy: WTI + Brent (links) & Natural Gas (rechts) ----------
have_energy = all(col in df.columns for col in ["wti_close", "brent_close", "natgas_close"])
energy = df[["date", "wti_close", "brent_close", "natgas_close"]].dropna(how="all") if have_energy else pd.DataFrame()

if have_energy and not energy.empty:
    e = energy.copy()
    e["wti_close"]    = _to_float(e["wti_close"])
    e["brent_close"]  = _to_float(e["brent_close"])
    e["natgas_close"] = _to_float(e["natgas_close"])

    fig_eng = make_subplots(specs=[[{"secondary_y": True}]])
    fig_eng.add_trace(go.Scatter(
        x=e["date"], y=e["wti_close"], name="WTI (USD/bbl)",
        line=dict(width=2, color=COLOR_WTI)
    ), secondary_y=False)
    fig_eng.add_trace(go.Scatter(
        x=e["date"], y=e["brent_close"], name="Brent (USD/bbl)",
        line=dict(width=2, color=COLOR_BRENT)
    ), secondary_y=False)
    fig_eng.add_trace(go.Scatter(
        x=e["date"], y=e["natgas_close"], name="NatGas (USD/MMBtu)",
        line=dict(width=2, color=COLOR_GAS)
    ), secondary_y=True)

    fig_eng.update_yaxes(title_text="Oil price (USD/bbl)", secondary_y=False)
    fig_eng.update_yaxes(title_text="Natural Gas (USD/MMBtu)", secondary_y=True)
    fig_eng.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        title="WTI & Brent vs Natural Gas",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_eng, use_container_width=True)
else:
    st.info("Energy-combo: ontbrekende kolommen of geen data (verwacht: wti_close, brent_close, natgas_close).")

# ---------- Metals: Gold (links) & Silver (rechts) ----------
have_metals = all(col in df.columns for col in ["gold_close", "silver_close"])
metals = df[["date", "gold_close", "silver_close"]].dropna(how="all") if have_metals else pd.DataFrame()

if have_metals and not metals.empty:
    m = metals.copy()
    m["gold_close"]   = _to_float(m["gold_close"])
    m["silver_close"] = _to_float(m["silver_close"])

    fig_met = make_subplots(specs=[[{"secondary_y": True}]])
    fig_met.add_trace(go.Scatter(
        x=m["date"], y=m["gold_close"], name="Gold (USD/oz)",
        line=dict(width=2, color=COLOR_GOLD)
    ), secondary_y=False)
    fig_met.add_trace(go.Scatter(
        x=m["date"], y=m["silver_close"], name="Silver (USD/oz)",
        line=dict(width=2, color=COLOR_SILVER)
    ), secondary_y=True)

<<<<<<< HEAD
    fig_met.update_yaxes(title_text="Gold (USD/oz)", secondary_y=False)
    fig_met.update_yaxes(title_text="Silver (USD/oz)", secondary_y=True)
    fig_met.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Gold vs Silver",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
=======
LABELS = {
    "wti": "WTI",
    "brent": "Brent",
    "gold": "Gold",
    "silver": "Silver",
    "gasoline": "Gasoline (RBOB)",
    "heatingoil": "Heating Oil",
    "natgas": "Natural Gas",
    "copper": "Copper",
}
def label_of(pfx: str) -> str:
    return LABELS.get(pfx, pfx.upper())

# ---- UI: filters & opties ----
min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    start, end = st.slider(
        "Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
>>>>>>> 3243f0a9da8ad473e524b7c4e7a276e03f93429f
    )
<<<<<<< HEAD
    st.plotly_chart(fig_met, use_container_width=True)
else:
    st.info("Metals-combo: ontbrekende kolommen of geen data (verwacht: gold_close, silver_close).")
=======
with c2:
    # standaard: ALLE instrumenten
    sel = st.multiselect(
        "Instrumenten (standaard alle)",
        options=[(p, label_of(p)) for p in prefixes],
        default=[(p, label_of(p)) for p in prefixes],
        format_func=lambda t: t[1],
    )
    sel = [p for p, _ in sel] or prefixes[:]  # fallback naar alle
with c3:
    avg_mode = st.radio("Gemiddelde", ["EMA", "SMA"], index=0, horizontal=True)
with c4:
    show_pairs = st.checkbox("Paarvergelijking (dubbele y-as)", value=False)
>>>>>>> 3243f0a9da8ad473e524b7c4e7a276e03f93429f

<<<<<<< HEAD
=======
st.caption("Elke selectie toont z’n eigen 2 grafieken: links prijs + MA/EMA 20/50/200, rechts Δ% bars + YTD% (secundaire as).")

# ---- Filter data ----
mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
df = df_wide.loc[mask].sort_values("date").copy()

# ---- Helpers ----
def cols_for(pfx: str) -> dict:
    return {
        "close": f"{pfx}_close",
        "d_abs": f"{pfx}_delta_abs",
        "d_pct": f"{pfx}_delta_pct",
        "ma20": f"{pfx}_ma20",
        "ma50": f"{pfx}_ma50",
        "ma200": f"{pfx}_ma200",
    }

def compute_ma_ema(series: pd.Series, kind: str, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "EMA":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    # SMA
    return s.rolling(window=window, min_periods=window).mean()

def ytd_percent_series(dates: pd.Series, values: pd.Series) -> pd.Series:
    """YTD% per jaar: (value / first_value_of_year - 1) * 100"""
    s = pd.Series(values.values, index=pd.to_datetime(dates), dtype="float64")
    years = s.index.year
    base = s.groupby(years).transform(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan)
    return (s / base - 1.0) * 100.0

# ---- Kleuren (Okabe-Ito) ----
COLOR_PRICE   = "#111111"  # bijna zwart
COLOR_MA20    = "#E69F00"  # oranje
COLOR_MA50    = "#009E73"  # groen
COLOR_MA200   = "#0072B2"  # blauw
COLOR_BAR_POS = "#009E73"  # groen bars
COLOR_BAR_NEG = "#D55E00"  # rood bars
COLOR_YTD     = "#CC79A7"  # paars

# ---- KPI's ----
st.subheader("Kerncijfers")
kpi_cols = st.columns(min(len(sel), 4) or 1)
for i, pfx in enumerate(sel or prefixes[:1]):
    c = cols_for(pfx)
    label = label_of(pfx)
    sub = df[["date", c["close"], c["d_abs"], c["d_pct"]]].dropna(subset=[c["close"]]).copy()
    with kpi_cols[i % len(kpi_cols)]:
        if sub.empty:
            st.metric(label, value="—", delta="—")
        else:
            last = sub.iloc[-1]
            val = float(last[c["close"]])
            d_abs = float(last[c["d_abs"]]) if pd.notnull(last[c["d_abs"]]) else 0.0
            d_pct = float(last[c["d_pct"]]) if pd.notnull(last[c["d_pct"]]) else 0.0
            st.metric(label, value=f"{val:,.2f}", delta=f"{d_abs:+.2f} ({d_pct*100:+.2f}%)")

st.markdown("---")

# ---- Per instrument: 2 grafieken naast elkaar (standaard voor ALLE selectie)
st.subheader("Per instrument")
for pfx in (sel or prefixes[:1]):
    c = cols_for(pfx)
    name = label_of(pfx)
    st.markdown(f"### {name}")
    left, right = st.columns(2)

    # Links: Close + MA/EMA 20/50/200
    needed = [c["close"]]
    if not all(col in df.columns for col in needed):
        with left:
            st.info("Geen prijsdata voor dit instrument.")
    else:
        sub = df[["date", c["close"]]].dropna().copy()
        sub[c["close"]] = pd.to_numeric(sub[c["close"]], errors="coerce").astype(float)

        # Bepaal MA/EMA: voor SMA gebruiken we databasekolommen als ze bestaan; anders berekenen
        if avg_mode == "SMA":
            ma20 = df[c["ma20"]] if c["ma20"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 20)
            ma50 = df[c["ma50"]] if c["ma50"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 50)
            ma200 = df[c["ma200"]] if c["ma200"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 200)
            # align op sub
            ma20 = pd.to_numeric(ma20, errors="coerce").astype(float).reindex(df.index).loc[sub.index]
            ma50 = pd.to_numeric(ma50, errors="coerce").astype(float).reindex(df.index).loc[sub.index]
            ma200 = pd.to_numeric(ma200, errors="coerce").astype(float).reindex(df.index).loc[sub.index]
        else:
            ma20 = compute_ma_ema(sub[c["close"]], "EMA", 20)
            ma50 = compute_ma_ema(sub[c["close"]], "EMA", 50)
            ma200 = compute_ma_ema(sub[c["close"]], "EMA", 200)

        with left:
            fig1 = make_subplots(specs=[[{"secondary_y": False}]])
            fig1.add_trace(go.Scatter(
                x=sub["date"], y=sub[c["close"]], name="Close",
                line=dict(width=2, color=COLOR_PRICE)
            ))
            if ma20.notna().any():
                fig1.add_trace(go.Scatter(
                    x=sub["date"], y=ma20.values, name=("EMA20" if avg_mode=="EMA" else "MA20"),
                    line=dict(width=2, color=COLOR_MA20)
                ))
            if ma50.notna().any():
                fig1.add_trace(go.Scatter(
                    x=sub["date"], y=ma50.values, name=("EMA50" if avg_mode=="EMA" else "MA50"),
                    line=dict(width=2, color=COLOR_MA50)
                ))
            if ma200.notna().any():
                fig1.add_trace(go.Scatter(
                    x=sub["date"], y=ma200.values, name=("EMA200" if avg_mode=="EMA" else "MA200"),
                    line=dict(width=2, color=COLOR_MA200)
                ))
            fig1.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig1, use_container_width=True)

    # Rechts: Δ% bars + YTD% (secundaire as)
    with right:
        if c["d_pct"] not in df.columns:
            st.info("Geen dagverandering beschikbaar voor dit instrument.")
        else:
            bars = df[["date", c["d_pct"]]].dropna().copy()
            bars[c["d_pct"]] = pd.to_numeric(bars[c["d_pct"]], errors="coerce").astype(float) * 100.0
            bar_colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in bars[c["d_pct"]].values]

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(
                go.Bar(x=bars["date"], y=bars[c["d_pct"]], name="Δ% per dag",
                       marker=dict(color=bar_colors), opacity=0.9),
                secondary_y=False
            )
            try:
                fig2.add_hline(y=0, line_dash="dot", opacity=0.5)
            except Exception:
                pass
            fig2.update_yaxes(title_text="Δ% dag", secondary_y=False)

            if c["close"] in df.columns:
                subc = df[["date", c["close"]]].dropna().copy()
                subc[c["close"]] = pd.to_numeric(subc[c["close"]], errors="coerce").astype(float)
                ytd = ytd_percent_series(subc["date"], subc[c["close"]])
                # align op bar-x
                ytd = ytd.reindex(pd.to_datetime(bars["date"]))
                fig2.add_trace(
                    go.Scatter(x=bars["date"], y=ytd.values, name="YTD%",
                               line=dict(width=2, color=COLOR_YTD)),
                    secondary_y=True
                )
                fig2.update_yaxes(title_text="YTD%", secondary_y=True)

            fig2.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---- Optionele paarvergelijkingen (dubbele y-as)
PAIR_CANDIDATES = [("wti", "brent"), ("gold", "silver"), ("gasoline", "heatingoil")]
if show_pairs:
    st.subheader("Paarvergelijkingen (dubbele y-as)")
    for a1, a2 in PAIR_CANDIDATES:
        if a1 not in prefixes or a2 not in prefixes:
            continue
        c1 = cols_for(a1); c2 = cols_for(a2)
        lbl1, lbl2 = label_of(a1), label_of(a2)
        sub = df[["date", c1["close"], c2["close"]]].dropna(how="all").copy()
        if sub.empty:
            continue
        for col in [c1["close"], c2["close"]]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce").astype(float)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if c1["close"] in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[c1["close"]], name=f"{lbl1} Close",
                line=dict(width=2, color=COLOR_PRICE)
            ), secondary_y=False)
        if c2["close"] in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[c2["close"]], name=f"{lbl2} Close",
                line=dict(width=2, color=COLOR_MA200)
            ), secondary_y=True)

        fig.update_yaxes(title_text=lbl1, secondary_y=False)
        fig.update_yaxes(title_text=lbl2, secondary_y=True)
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            title=f"{lbl1} vs {lbl2}"
        )
        st.plotly_chart(fig, use_container_width=True)

>>>>>>> 3243f0a9da8ad473e524b7c4e7a276e03f93429f