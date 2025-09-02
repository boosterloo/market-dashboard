# utils/nav.py
import streamlit as st
from typing import List, Optional

def _safe_page_link(page: str, label: str):
    """
    Try st.page_link; on any failure, fall back to switch_page or a markdown link.
    """
    has_page_link = hasattr(st.sidebar, "page_link") or hasattr(st, "page_link")
    has_switch_page = hasattr(st, "switch_page")

    # 1) Preferred: native page_link
    if has_page_link:
        try:
            # Streamlit >= 1.30
            st.sidebar.page_link(page=page, label=label, icon=None)
            return
        except Exception:
            # Fall through to button fallback
            pass

    # 2) Fallback: button + st.switch_page
    cols = st.sidebar.columns([1, 6])
    with cols[1]:
        if has_switch_page:
            if st.button(f"➡️ {label}", key=f"navbtn::{label}::{page}"):
                try:
                    st.switch_page(page)
                except Exception:
                    # 3) Final fallback: a plain link the user can click
                    st.sidebar.markdown(f"- [{label}]({page})")
        else:
            # 3) Final fallback: plain link
            st.markdown(f"- [{label}]({page})")

def sidebar_nav(
    entries: List[dict],
    section_title: str = "📚 Dashboard",
    highlight_current: bool = True,
    current_slug: Optional[str] = None,
):
    """
    entries: list of dicts with keys:
      - label: str (visible name)
      - page:  str (script path exactly as Streamlit registers it, e.g. 'pages/4_Greeks_3D.py' or 'streamlit_app.py')
      - icon:  Optional[str] (emoji)
    """
    st.sidebar.markdown(f"### {section_title}")

    for e in entries:
        label = e.get("label", "Page")
        page  = e.get("page", "")
        icon  = e.get("icon", "")

        # Skip empty/placeholder items cleanly
        if not page:
            continue

        shown_label = f"{icon} {label}" if icon else label
        # Optional visual hint for current page
        if highlight_current and current_slug and (current_slug.lower() in label.lower()):
            shown_label = f"**{shown_label}**"

        _safe_page_link(page, shown_label)

    st.sidebar.markdown("---")
