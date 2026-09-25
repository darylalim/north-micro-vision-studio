"""North Micro Vision Studio — entry point and router.

Run with:  uv run streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="North Micro Vision Studio",
    page_icon=":material/frame_inspect:",
    layout="wide",
)

# ty >= 0.0.82 resolves `st.navigation` to the `streamlit.navigation` submodule:
# streamlit/__init__.py binds the function, then imports streamlit.navigation.page
# on the next line, and ty now treats that import as rebinding the name. At
# runtime the submodule is already loaded by then, so Python leaves the function
# in place — this is the function, and the call is sound.
st.navigation(  # ty: ignore[call-non-callable]
    [
        st.Page("app_pages/chat.py", title="Chat", icon=":material/forum:", default=True),
        st.Page(
            "app_pages/grounding.py",
            title="Grounding",
            icon=":material/frame_inspect:",
        ),
    ]
).run()
