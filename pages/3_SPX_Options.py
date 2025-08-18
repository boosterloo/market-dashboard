# ── B) PPD & Afstand tot Uitoefenprijs ─────────────────────────────────────────
st.subheader("PPD & Afstand tot Uitoefenprijs (ATM→OTM/ITM)")

# ▼ Kies peildatum (default: meest recente)
snapshots = sorted(df["snapshot_date"].dt.floor("min").unique())
default_idx = len(snapshots) - 1 if snapshots else 0
sel_snapshot = st.selectbox(
    "Peildatum (snapshot)", options=snapshots, index=default_idx,
    format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
)

df_last = df[df["snapshot_date"] == sel_snapshot].copy()

# ▼ Outlier‑handling opties
st.caption("Outliers: beïnvloeden schaal en zichtbaarheid van PPD/Price")
col_out1, col_out2 = st.columns([1.2, 1])
with col_out1:
    outlier_mode = st.radio(
        "Outlier-methode",
        ["Geen", "Percentiel clip", "IQR filter", "Z‑score filter"],
        horizontal=True, index=1
    )
with col_out2:
    pct_clip = st.slider("Percentiel clip (links/rechts)", 0, 10, 1, step=1, help="Alleen gebruikt bij 'Percentiel clip'")

def _apply_outlier(df_series: pd.Series) -> pd.Series:
    s = df_series.astype(float)
    if outlier_mode == "Geen":
        return s
    if outlier_mode == "Percentiel clip":
        lo, hi = np.nanpercentile(s, [pct_clip, 100 - pct_clip]) if s.notna().any() else (0, 1)
        return s.clip(lower=lo, upper=hi)
    if outlier_mode == "IQR filter":
        q1, q3 = np.nanpercentile(s, [25, 75]) if s.notna().any() else (0, 1)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return s.where((s >= lo) & (s <= hi), np.nan)
    if outlier_mode == "Z‑score filter":
        mu, sd = np.nanmean(s), np.nanstd(s)
        if sd == 0 or np.isnan(sd):
            return s
        z = (s - mu) / sd
        return s.where(np.abs(z) <= 3.0, np.nan)
    return s

# --- PPD vs afstand (op peildatum) ---
ppd_vs_dist = (
    df_last.assign(ppd_f=_apply_outlier(df_last["ppd"]))
          .groupby(((df_last["dist_points"].abs() / df_last["underlying_price"]) * 100.0)
          .round(2), as_index=False)["ppd_f"].mean()
          .rename(columns={"ppd_f": "ppd"})
          .sort_values(0)
)
ppd_vs_dist.rename(columns={0: "abs_dist_pct"}, inplace=True)

fig_ppd_dist = go.Figure(go.Scatter(
    x=ppd_vs_dist["abs_dist_pct"], y=ppd_vs_dist["ppd"],
    mode="lines+markers", name="PPD"
))
fig_ppd_dist.update_layout(
    title=f"PPD vs Afstand tot Strike — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
    xaxis_title="Afstand tot strike (|strike − underlying| / underlying, %)",
    yaxis_title="PPD",
    height=400
)
st.plotly_chart(fig_ppd_dist, use_container_width=True)

# ── C) Ontwikkeling Prijs per Expiratiedatum (peildatum) ───────────────────────
st.subheader("Ontwikkeling Prijs per Expiratiedatum (laatste snapshot)")

exp_curve_raw = (
    df_last[df_last["strike"] == series_strike]
      .groupby("expiration", as_index=False)
      .agg(price=("last_price", "mean"), ppd=("ppd", "mean"))
      .sort_values("expiration")
)

# outlier‑handling per serie
exp_curve = exp_curve_raw.copy()
exp_curve["price_f"] = _apply_outlier(exp_curve["price"])
exp_curve["ppd_f"]   = _apply_outlier(exp_curve["ppd"])

fig_exp = make_subplots(specs=[[{"secondary_y": True}]])
fig_exp.add_trace(
    go.Scatter(x=exp_curve["expiration"], y=exp_curve["price_f"],
               name="Price", mode="lines+markers"),
    secondary_y=False
)
fig_exp.add_trace(
    go.Scatter(x=exp_curve["expiration"], y=exp_curve["ppd_f"],
               name="PPD", mode="lines+markers"),
    secondary_y=True
)

# Robuuste autoscale per as (onafhankelijk) en netjes vanaf 0
fig_exp.update_layout(
    title=f"{sel_type.upper()} — Strike {series_strike} — peildatum {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
    height=420, hovermode="x unified"
)
fig_exp.update_xaxes(title_text="Expiratiedatum")
fig_exp.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
fig_exp.update_yaxes(title_text="PPD",   secondary_y=True,  rangemode="tozero")

# hint tonen als we hebben ingeclipt of gefilterd
if outlier_mode != "Geen":
    fig_exp.add_annotation(
        xref="paper", yref="paper", x=1, y=1.08, xanchor="right", showarrow=False,
        text=f"Outlier-methode: {outlier_mode}"
    )

st.plotly_chart(fig_exp, use_container_width=True)
