import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)


# ============================================================
# PAGE NAVIGATION
# ============================================================

detector_page = st.Page(
    "detector.py",
    title="Spam Detector",
    icon="🏠"
)

dashboard_page = st.Page(
    "dashboard.py",
    title="Dashboard",
    icon="📊"
)


# ============================================================
# NAVIGATION
# ============================================================

pg = st.navigation(
    [detector_page, dashboard_page],
    position="top"
)

pg.run()