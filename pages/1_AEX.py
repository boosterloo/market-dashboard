

def build_summary(row):
    trend = classify_trend(row)
    strength = classify_trend_strength(row)
    momentum = classify_momentum(row)
    exhaustion = classify_exhaustion(row)
    vol = classify_vol_regime(row)

    parts = []

    if trend in ["Strong Bull", "Bull"]:
        parts.append(f"Trend is {trend.lower()} met {strength.lower()} trendkracht")
    elif trend in ["Strong Bear", "Bear"]:
        parts.append(f"Trend is {trend.lower()} met {strength.lower()} trendkracht")
    else:
        parts.append("Markt zit meer in een neutrale of overgangsfase")

    if momentum != "Onvoldoende data":
        parts.append(f"momentum oogt {momentum.lower()}")

    if exhaustion == "Hoog":
        parts.append("uitputting is hoog")
    elif exhaustion == "Oplopend":
        parts.append("uitputting loopt op")
    else:
        parts.append("uitputting blijft beperkt")

    if vol != "Onvoldoende data":
        parts.append(f"volatiliteitsregime staat op {vol.lower()}")

    return ". ".join(parts) + "."


d["macd_hist_prev"] = d["macd_hist"].shift(1)
d["trend_label"] = d.apply(classify_trend, axis=1)
d["trend_strength"] = d.apply(classify_trend_strength, axis=1)
d["momentum_label"] = d.apply(classify_momentum, axis=1)
d["exhaustion_label"] = d.apply(classify_exhaustion, axis=1)
d["vol_regime"] = d.apply(classify_vol_regime, axis=1)

state_ready = d.dropna(subset=["close"]).copy()
last_state_row = state_ready.iloc[-1] if not state_ready.empty else d.iloc[-1]

# =========================
# Regime shading helper
# =========================
def add_regime_spans(fig, data, row=1, col=1):
    colors = {
        "Strong Bull": "rgba(0,140,70,0.08)",
        "Bull": "rgba(0,180,90,0.05)",
        "Neutral": "rgba(130,130,130,0.05)",
        "Bear": "rgba(220,90,0,0.05)",
        "Strong Bear": "rgba(200,0,0,0.08)",
    }
    lbl = data["trend_label"].fillna("Neutral")
    if lbl.empty:
        return

    start_idx = 0
    current = lbl.iloc[0]

    for i in range(1, len(lbl)):
        if lbl.iloc[i] != current:
            fig.add_vrect(
                x0=data["date"].iloc[start_idx],
                x1=data["date"].iloc[i - 1],
                fillcolor=colors.get(current, "rgba(130,130,130,0.04)"),
                line_width=0,
                row=row,
                col=col,
                layer="below",
            )
            start_idx = i
            current = lbl.iloc[i]

    fig.add_vrect(
        x0=data["date"].iloc[start_idx],
        x1=data["date"].iloc[len(lbl) - 1],
        fillcolor=colors.get(current, "rgba(130,130,130,0.04)"),
        line_width=0,
        row=row,
        col=col,
        layer="below",
    )


# =========================
# Top diagnostics
# =========================
summary_text = build_summary(last_state_row)
ytd_full = ytd_return_full(df)

last_close_val = safe_last(d["close"])
last_delta_day = safe_last(d["close"].pct_change() * 100.0)
last_vix_val = safe_last(d["vix_close"]) if "vix_close" in d.columns else np.nan
last_rsi_val = safe_last(d["rsi14_s"])
last_adx_val = safe_last(d["adx14"])
last_macd_hist_val = safe_last(d["macd_hist"])
last_stretch_val = safe_last(d["stretch_ema20_atr"])
last_z20_val = safe_last(d["z20"])
last_atr_pct_val = safe_last(d["atr_pct_close"])

k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
k1.metric("Laatste close", fmt_num(last_close_val))
k2.metric("Delta % dag", fmt_num(last_delta_day, 2, "%"))
k3.metric("Trend", last_state_row["trend_label"])
k4.metric("Trendkracht", last_state_row["trend_strength"])
k5.metric("Momentum", last_state_row["momentum_label"])
k6.metric("Uitputting", last_state_row["exhaustion_label"])
k7.metric("Vol-regime", last_state_row["vol_regime"])
k8.metric("YTD", fmt_num(ytd_full, 2, "%") if ytd_full is not None else "-")
k9.metric("VIX", fmt_num(last_vix_val))

st.info(summary_text)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("RSI(14) smoothed", fmt_num(last_rsi_val))
m2.metric("ADX(14)", fmt_num(last_adx_val))
m3.metric("MACD hist", fmt_num(last_macd_hist_val, 3))
m4.metric("Stretch vs EMA20", fmt_num(last_stretch_val, 2, " ATR"))
m5.metric("Z-score 20d", fmt_num(last_z20_val, 2))
m6.metric("ATR % close", fmt_num(last_atr_pct_val, 2, "%"))

# =========================
# Main chart: Heikin Ashi
# =========================
fig1 = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"secondary_y": True}]],
    subplot_titles=["AEX Heikin Ashi + EMA20/50/200" + (" + VIX" if show_vix else "")]
)

if show_regime_shading:
    add_regime_spans(fig1, d, row=1, col=1)

ha_plot = d.dropna(subset=["ha_open", "ha_high", "ha_low", "ha_close"]).copy()

fig1.add_trace(
    go.Candlestick(
        x=ha_plot["date"],
        open=ha_plot["ha_open"],
        high=ha_plot["ha_high"],
        low=ha_plot["ha_low"],
        close=ha_plot["ha_close"],
        name="Heikin Ashi"
    ),
    row=1, col=1, secondary_y=False
)

for span in DEFAULTS["ema_spans"]:
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d[f"ema{span}"], mode="lines", name=f"EMA{span}", line=dict(width=2)),
        row=1, col=1, secondary_y=False
    )

if show_donchian:
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d["dc_high"], mode="lines", name="DC High", line=dict(width=1, dash="dot")),
        row=1, col=1, secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d["dc_low"], mode="lines", name="DC Low", line=dict(width=1, dash="dot")),
        row=1, col=1, secondary_y=False
    )

if show_vix and "vix_close" in d.columns and d["vix_close"].notna().any():
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d["vix_close"], mode="lines", name="VIX", line=dict(width=2)),
        row=1, col=1, secondary_y=True
    )

fig1.update_layout(
    height=700,
    margin=dict(l=60, r=60, t=80, b=40),
    legend_orientation="h",
    legend_yanchor="top",
    legend_y=1.08,
    legend_x=0,
    xaxis_rangeslider_visible=False,
)
fig1.update_xaxes(tickfont=dict(size=13))
fig1.update_yaxes(title_text="AEX", row=1, col=1, tickfont=dict(size=13), secondary_y=False)
fig1.update_yaxes(title_text="VIX", row=1, col=1, tickfont=dict(size=13), secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# =========================
# Momentum dashboard
# =========================
fig2 = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "RSI(14) Wilder + dynamische zones",
        "MACD(12,26,9)",
        "ADX(14) + DI",
        "Stretch vs EMA20 in ATR",
    ],