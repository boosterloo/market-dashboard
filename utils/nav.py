# utils/nav.py
import streamlit as st
from typing import List, Optional

def sidebar_nav(
    entries: List[dict],
    section_title: str = "📚 Dashboard",
    highlight_current: bool = True,
    current_slug: Optional[str] = None,
):
    """
    entries: lijst van dicts met keys:
      - label: str (zichtbare naam)
      - page:  str (pad, bv. 'pages/3_SPX_Options.py' of 'Home')
      - icon:  Optional[str] (emoji)
    current_slug: (optioneel) slug/naam om actief item te highlighten (bv. 'SPX Options')

    Werkt met:
      - st.page_link (als beschikbaar)  ✅
      - st.switch_page (als beschikbaar) bij clicks  ✅
      - markdown-links fallback  ✅
    """

    st.sidebar.markdown(f"### {section_title}")

    has_page_link = hasattr(st.sidebar, "page_link") or hasattr(st, "page_link")
    has_switch_page = hasattr(st, "switch_page")

    for e in entries:
        label = e.get("label", "Page")
        page  = e.get("page", "")
        icon  = e.get("icon", "")

        shown_label = f"{icon} {label}" if icon else label
        is_current = (highlight_current and current_slug and (current_slug.lower() in label.lower()))

        if has_page_link:
            # Streamlit >= 1.30: page_link in de sidebar beschikbaar
            st.sidebar.page_link(page=page, label=shown_label, icon=None)
        else:
            # Fallback: render een knop + link
            # Knop -> switch_page als beschikbaar; anders een markdown link
            cols = st.sidebar.columns([1, 5])
            with cols[1]:
                btn_label = f"➡️ {label}" if is_current else label
                if has_switch_page:
                    if st.button(btn_label, key=f"navbtn::{label}"):
                        try:
                            st.switch_page(page)
                        except Exception:
                            pass
                else:
                    # Markdown link fallback – toont gewoon een link naar de page (werkt prima in multipage apps)
                    st.markdown(f"- [{shown_label}]({page})", help="Pagina-link")

    st.sidebar.markdown("---")
