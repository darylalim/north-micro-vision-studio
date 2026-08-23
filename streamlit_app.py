"""North Micro Vision Studio — entry point and router.

Run with:  uv run streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="North Micro Vision Studio",
    page_icon=":material/frame_inspect:",
    layout="wide",
)

st.navigation(
    [
        st.Page("app_pages/chat.py", title="Chat", icon=":material/forum:", default=True),
        st.Page(
            "app_pages/grounding.py",
            title="Grounding",
            icon=":material/frame_inspect:",
        ),
    ]
).run()
