
from datetime import date, timedelta
import streamlit as st
from utils.bq import load_vix
from utils.helpers import ensure_datetime
from utils.charts import plot_line

st.set_page_config(page_title="VIX", layout="wide")

st.sidebar.subheader("📚 Pagina's")
st.sidebar.page_link("app.py", label="🏠 Dashboard instellingen")
st.sidebar.page_link("pages/1_SP500.py", label="📈 S&P 500 + VIX")
st.sidebar.page_link("pages/2_VIX.py", label="📉 VIX (alleen)")
st.sidebar.page_link("pages/3_SPX_Options.py", label="🧮 SPX Opties")

st.title("📉 VIX (alleen)")

view_start = st.session_state.get("view_start")
view_end   = st.session_state.get("view_end")

if not (view_start and view_end):
    view_end = date.today()
    view_start = view_end - timedelta(days=90)

df_vix = load_vix(view_start, view_end)
if df_vix.empty:
    st.warning("Geen VIX data in de gekozen periode.")
else:
    df_vix = ensure_datetime(df_vix, date_col="date")
    fig = plot_line(
        df=df_vix,
        x="date",
        y_cols=["close"],
        names=["VIX Close"],
        title="VIX — Close",
        show_volume=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Tabel VIX (download)"):
        st.dataframe(df_vix)
        st.download_button(
            "Download CSV (VIX)",
            data=df_vix.to_csv(index=False).encode("utf-8"),
            file_name="vix.csv",
            mime="text/csv",
        )
