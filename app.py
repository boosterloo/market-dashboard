import streamlit as st
st.set_page_config(page_title="app", layout="wide")
st.sidebar.title("app")
st.write("Kies links een pagina.")
# Sidebar quick links (automatisch)
import os
if os.path.isdir("pages"):
    st.sidebar.markdown("### 📚 Pagina's")
    for fname in sorted(os.listdir("pages")):
        if fname.endswith(".py"):
            label = fname.split(".py")[0].replace("_", " ")
            st.sidebar.page_link(f"./pages/{fname}", label=label)
