    return f"{x:.{nd}f}{suffix}"


def ytd_return_full(full_df: pd.DataFrame):
    sub = full_df.dropna(subset=["date", "close"]).copy()
    if sub.empty:
        return None
    max_d = sub["date"].max()
    start = pd.Timestamp(date(max_d.year, 1, 1))
    sub = sub[sub["date"] >= start]
    return (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100 if len(sub) >= 2 else None


def pytd_return_full(full_df: pd.DataFrame):
    sub = full_df.dropna(subset=["date", "close"]).copy()
    if sub.empty:
        return None
    max_d = sub["date"].max()
    prev_year = max_d.year - 1
    start = pd.Timestamp(date(prev_year, 1, 1))
    try:
        end = max_d.replace(year=prev_year)
    except Exception:
        end = pd.Timestamp(date(prev_year, 12, 31))
    sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
    return (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100 if len(sub) >= 2 else None


def heikin_ashi(src: pd.DataFrame):
    ha = src.copy()
    ha["ha_close"] = (ha["open"] + ha["high"] + ha["low"] + ha["close"]) / 4.0

    ha_open = pd.Series(index=ha.index, dtype=float)
    first_valid_idx = ha[["open", "close"]].dropna().index.min()

    if pd.isna(first_valid_idx):
        ha["ha_open"] = np.nan
        ha["ha_high"] = np.nan
        ha["ha_low"] = np.nan
        return ha[["ha_open", "ha_high", "ha_low", "ha_close"]]

    ha_open.loc[first_valid_idx] = (ha.loc[first_valid_idx, "open"] + ha.loc[first_valid_idx, "close"]) / 2.0
    start_pos = ha.index.get_loc(first_valid_idx)

    for i in range(start_pos + 1, len(ha)):
        prev_idx = ha.index[i - 1]
        cur_idx = ha.index[i]
        if pd.notna(ha_open.loc[prev_idx]) and pd.notna(ha.loc[prev_idx, "ha_close"]):
            ha_open.loc[cur_idx] = (ha_open.loc[prev_idx] + ha.loc[prev_idx, "ha_close"]) / 2.0
        else:
            ha_open.loc[cur_idx] = np.nan

    ha["ha_open"] = ha_open
    ha["ha_high"] = pd.concat([ha["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"] = pd.concat([ha["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)

    return ha[["ha_open", "ha_high", "ha_low", "ha_close"]]


# =========================
# Indicator calculation
# =========================
@st.cache_data(ttl=1800)
def compute_indicators(full_df: pd.DataFrame):
    dfx = full_df.copy()

    for span in DEFAULTS["ema_spans"]:
        dfx[f"ema{span}"] = ema(dfx["close"], span)

    dfx["atr14"] = atr_rma(dfx["high"], dfx["low"], dfx["close"], 14)
    dfx["macd_line"], dfx["macd_signal"], dfx["macd_hist"] = macd(dfx["close"], *DEFAULTS["macd"])
    dfx["di_plus"], dfx["di_minus"], dfx["adx14"] = adx(dfx, DEFAULTS["adx_length"])

    dfx["rsi14"] = rsi_wilder(dfx["close"], DEFAULTS["rsi_period"])
    dfx["rsi14_s"] = dfx["rsi14"].ewm(span=5, adjust=False).mean()
    dfx["rsi_dyn_hi"] = rolling_percentile(dfx["rsi14"], 0.80, DEFAULTS["rsi_dyn_win"])
    dfx["rsi_dyn_lo"] = rolling_percentile(dfx["rsi14"], 0.20, DEFAULTS["rsi_dyn_win"])

    dfx["dc_high"], dfx["dc_low"] = donchian(dfx, DEFAULTS["donchian_n"])
    dfx["ema20_slope_10"] = slope_pct(dfx["ema20"], 10)
    dfx["ema50_slope_10"] = slope_pct(dfx["ema50"], 10)

    dfx["stretch_ema20_atr"] = (dfx["close"] - dfx["ema20"]) / dfx["atr14"].replace(0, np.nan)
    dfx["stretch_ema50_atr"] = (dfx["close"] - dfx["ema50"]) / dfx["atr14"].replace(0, np.nan)
    dfx["z20"] = zscore(dfx["close"], 20)
    dfx["z50"] = zscore(dfx["close"], 50)
    dfx["atr_pct_close"] = (dfx["atr14"] / dfx["close"]) * 100.0

    dfx["rv_10"] = dfx["close"].pct_change().rolling(10).std() * np.sqrt(252) * 100.0
    dfx["rv_20"] = dfx["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100.0

    if "vix_close" in dfx.columns:
        vix_ma20 = dfx["vix_close"].rolling(20, min_periods=20).mean()
        vix_sd20 = dfx["vix_close"].rolling(20, min_periods=20).std()
        dfx["vix_z"] = (dfx["vix_close"] - vix_ma20) / vix_sd20.replace(0, np.nan)
        dfx["vix_change_5d"] = dfx["vix_close"].pct_change(5) * 100.0
        dfx["vix_rv20_spread"] = dfx["vix_close"] - dfx["rv_20"]
    else:
        dfx["vix_close"] = np.nan
        dfx["vix_z"] = np.nan
        dfx["vix_change_5d"] = np.nan
        dfx["vix_rv20_spread"] = np.nan

    ha = heikin_ashi(dfx)
    dfx[["ha_open", "ha_high", "ha_low", "ha_close"]] = ha[["ha_open", "ha_high", "ha_low", "ha_close"]]

    dfx["delta_abs"] = dfx["delta_abs"].fillna(0)
    dfx["delta_pct"] = dfx["delta_pct"].fillna(0)

    return dfx


df = compute_indicators(df)

# =========================
# Periode + sidebar
# =========================
min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max((df["date"].max() - timedelta(days=365)).date(), min_d)

c1, c2, c3 = st.columns([0.08, 0.84, 0.08])
with c2:
    start_date, end_date = st.slider(
        "Periode",
        min_value=min_d,
        max_value=max_d,
        value=(default_start, max_d),
        format="YYYY-MM-DD",
    )

with st.sidebar:
    st.markdown("### Instellingen")

    st.markdown("#### Grafieken")
    show_vix = st.toggle("Toon VIX in hoofdgrafiek", value=True)
    show_donchian = st.toggle("Toon Donchian", value=True)
    show_regime_shading = st.toggle("Toon regime shading", value=True)

    st.divider()
    st.markdown("#### Delta")
    delta_mode = st.radio("Weergave", ["Delta punten", "Delta %"], index=0)
    agg_mode = st.selectbox("Aggregatie", ["Dagelijks", "Wekelijks", "Maandelijks"], index=0)
    smooth_on = st.checkbox("Smoothing MA", value=False)
    ma_window = st.slider("MA-window", 2, 60, 5, step=1, disabled=not smooth_on)

    st.divider()
    st.markdown("#### Correlatie")
    corr_vs = st.radio("Rolling correlatie vs VIX", ["% change", "level"], index=0)
    corr_win = st.slider("Correlatie-window", 5, 90, DEFAULTS["corr_win_default"], step=1)

    st.divider()
    st.markdown("#### Forward returns")
    future_horizons = st.multiselect(
        "Horizons",
        options=[1, 3, 5, 10, 20],
        default=[1, 3, 5, 10],
    )

# =========================
# Filter subset
# =========================
d = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].reset_index(drop=True).copy()

if d.empty or len(d) < 30:
    st.warning("Te weinig data in de gekozen periode.")
    st.stop()

# =========================
# State engine
# =========================
def classify_trend(row):
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["ema200"]):
        return "Onvoldoende data"

    c = row["close"]
    e20 = row["ema20"]
    e50 = row["ema50"]
    e200 = row["ema200"]
    s20 = row["ema20_slope_10"]
    s50 = row["ema50_slope_10"]

    if c > e20 > e50 > e200 and s20 > 0 and s50 > 0:
        return "Strong Bull"
    if c > e50 > e200:
        return "Bull"
    if c < e20 < e50 < e200 and s20 < 0 and s50 < 0:
        return "Strong Bear"
    if c < e50 < e200:
        return "Bear"
    return "Neutral"


def classify_trend_strength(row):
    adx_value = row["adx14"]
    spread1 = abs((row["ema20"] / row["ema50"] - 1) * 100) if pd.notna(row["ema20"]) and pd.notna(row["ema50"]) and row["ema50"] != 0 else np.nan
    spread2 = abs((row["ema50"] / row["ema200"] - 1) * 100) if pd.notna(row["ema50"]) and pd.notna(row["ema200"]) and row["ema200"] != 0 else np.nan