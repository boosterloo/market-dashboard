
from datetime import date, timedelta
import streamlit as st

st.set_page_config(page_title="Market Dashboard", layout="wide")

# Sidebar
st.sidebar.subheader("📚 Pagina's")
st.sidebar.page_link("app.py", label="🏠 Dashboard instellingen")
st.sidebar.page_link("pages/1_SP500.py", label="📈 S&P 500 + VIX")
st.sidebar.page_link("pages/2_VIX.py", label="📉 VIX (alleen)")
st.sidebar.page_link("pages/3_SPX_Options.py", label="🧮 SPX Opties")

st.title("📊 Market Dashboard — Instellingen")

DEFAULT_END = date.today()
DEFAULT_START = DEFAULT_END - timedelta(days=120)

periode = st.date_input(
    "Periode",
    value=(st.session_state.get("period_start", DEFAULT_START),
           st.session_state.get("period_end", DEFAULT_END)),
    format="YYYY-MM-DD",
)

if isinstance(periode, tuple) and len(periode) == 2:
    period_start, period_end = periode
else:
    period_start = periode
    period_end = DEFAULT_END

slider_start, slider_end = st.slider(
    "Zoom binnen gekozen periode",
    min_value=period_start,
    max_value=period_end,
    value=(
        st.session_state.get("view_start", period_start),
        st.session_state.get("view_end", period_end),
    ),
    help="Gebruik deze slider om snel binnen de opgehaalde periode te zoomen.",
)

st.session_state["period_start"] = period_start
st.session_state["period_end"] = period_end
st.session_state["view_start"] = slider_start
st.session_state["view_end"] = slider_end

st.success(
    f"Actieve weergave: {slider_start.isoformat()} → {slider_end.isoformat()}  "
    f"(binnen {period_start.isoformat()} → {period_end.isoformat()})"
)

st.info("Ga in de sidebar naar een pagina om de visualisaties te bekijken.")
