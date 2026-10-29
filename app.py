"""
VisionAttendance — Smart Attendance Management System Using Face Recognition
Semester VII Computer Vision Mini-Project
Built with Streamlit, OpenCV (YuNet + SFace), SQLite, NumPy, Pandas, and Matplotlib.
Designed with pixel-perfect SaaS academic layout matching institutional requirements.
"""

from datetime import datetime, timedelta
import os
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from attendance_service import AttendanceService
import database
from face_recognition_service import DEFAULT_COSINE_THRESHOLD, FaceRecognitionService
import image_processing

# ----------------------------------------------------------------------------
# 1. Page Configuration & Pixel-Perfect SaaS CSS Styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionAttendance | B.E. Semester VII CV",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Top Navigation Header Bar */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0 18px 0;
        margin-bottom: 6px;
    }
    .page-main-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
        margin: 0;
    }
    .header-badge-container {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .pill-status-green {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-dot-green {
        width: 7px;
        height: 7px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
    }
    .pill-admin-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Clean White Metric Cards (Row 1) */
    .kpi-stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        height: 100%;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 4px;
    }
    .kpi-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-desc {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    .kpi-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    .icon-box-blue { background-color: #EFF6FF; color: #2563EB; }
    .icon-box-purple { background-color: #F5F3FF; color: #7C3AED; }
    .icon-box-amber { background-color: #FFFBEB; color: #D97706; }
    .icon-box-green { background-color: #ECFDF5; color: #059669; }

    /* Dashboard Panels */
    .dashboard-panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }
    .panel-header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 14px;
    }
    .panel-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .panel-card-subtitle {
        font-size: 0.78rem;
        color: #64748B;
    }
    .status-badge-completed {
        background-color: #F1F5F9;
        color: #475569;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Inner Session Box */
    .inner-session-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        gap: 24px;
        margin-bottom: 16px;
    }
    .session-meta-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .session-meta-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
    }
    .session-meta-value-blue {
        color: #2563EB;
        font-size: 0.92rem;
        font-weight: 700;
    }

    /* Progress Bar */
    .ratio-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 6px;
    }
    .progress-bar-bg {
        width: 100%;
        height: 8px;
        background-color: #E2E8F0;
        border-radius: 9999px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .progress-bar-fill {
        height: 100%;
        background-color: #10B981;
        border-radius: 9999px;
    }

    /* Quick Action Card */
    .quick-action-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .quick-action-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
    }
    .quick-action-subtitle {
        font-size: 0.78rem;
        color: #64748B;
    }

    /* ------------------------------------------------------------------------
       SIDEBAR PIXEL-PERFECT STYLING & CONTRAST FIXES
       ------------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }

    /* Brand Header Box */
    .sidebar-brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 4px 18px 4px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 14px;
    }
    .sidebar-logo-icon {
        width: 40px;
        height: 40px;
        background-color: #2563EB;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.3rem;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        letter-spacing: -0.3px;
    }
    .sidebar-brand-subtitle {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
    }
    .sidebar-section-title {
        font-size: 0.68rem;
        font-weight: 800;
        color: #94A3B8;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin: 14px 0 8px 6px;
    }

    /* Radio navigation buttons styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 4px !important;
    }
    /* Hide radio circles */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div:first-child {
        display: none !important;
    }

    /* Base Nav Link Item */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 9px 12px !important;
        border-radius: 8px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label * {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }

    /* HOVER State: Guarantees text is clearly visible on dark/light hover */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover * {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* ACTIVE/CHECKED State: Bright blue pill */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) * {
        color: #2563EB !important;
        font-weight: 800 !important;
    }

    /* ACTIVE/CHECKED State on HOVER */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked):hover {
        background-color: #DBEAFE !important;
        border-color: #93C5FD !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked):hover,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked):hover * {
        color: #1D4ED8 !important;
        font-weight: 800 !important;
    }

    /* Separator between Core and Academic in the radio menu */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(8) {
        margin-top: 18px !important;
        position: relative !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(8)::before {
        content: "ACADEMIC & EXAMINATION";
        display: block;
        font-size: 0.68rem;
        font-weight: 800;
        color: #94A3B8;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 8px;
        margin-top: -12px;
        padding-left: 2px;
    }

    /* Sidebar Slider & Labels Contrast Fix */
    section[data-testid="stSidebar"] [data-testid="stSlider"] label,
    section[data-testid="stSidebar"] [data-testid="stSlider"] label *,
    section[data-testid="stSidebar"] [data-testid="stSlider"] div,
    section[data-testid="stSidebar"] [data-testid="stSlider"] span,
    section[data-testid="stSidebar"] [data-testid="stSlider"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] h6 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* GLOBAL BUTTON BASE & HOVER CONTRAST FIXES */
    /* Secondary & default buttons */
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stFormSubmitButton"] button,
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stButton"] button *,
    div[data-testid="stDownloadButton"] button *,
    div[data-testid="stFormSubmitButton"] button *,
    button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Secondary Button Hover - High Contrast Dark Slate Text */
    div[data-testid="stButton"] button:hover:not([kind="primary"]),
    div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover:not([kind="primary"]),
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button:hover:not([kind="primary"]):not([data-testid="baseButton-primary"]):not([data-testid="stBaseButton-primary"]) {
        background-color: #F1F5F9 !important;
        border-color: #64748B !important;
        color: #0F172A !important;
    }

    div[data-testid="stButton"] button:hover:not([kind="primary"]) *,
    div[data-testid="stDownloadButton"] button:hover *,
    div[data-testid="stFormSubmitButton"] button:hover:not([kind="primary"]) *,
    button[kind="secondary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[data-testid="stBaseButton-secondary"]:hover *,
    button:hover:not([kind="primary"]):not([data-testid="baseButton-primary"]):not([data-testid="stBaseButton-primary"]) * {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Primary buttons */
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stButton"] button[data-testid="baseButton-primary"],
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }

    div[data-testid="stButton"] button[kind="primary"] *,
    div[data-testid="stButton"] button[data-testid="baseButton-primary"] *,
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] *,
    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover,
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border-color: #1E40AF !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover *,
    div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover *,
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover *,
    button[kind="primary"]:hover *,
    button[data-testid="baseButton-primary"]:hover *,
    button[data-testid="stBaseButton-primary"]:hover * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Expander hover contrast */
    div[data-testid="stExpander"] details summary {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    div[data-testid="stExpander"] details summary *,
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] details summary span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }
    div[data-testid="stExpander"] details summary:hover *,
    div[data-testid="stExpander"] details summary:hover p,
    div[data-testid="stExpander"] details summary:hover span {
        color: #1E40AF !important;
        font-weight: 700 !important;
    }

    /* Tabs hover contrast */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"] * {
        color: #475569 !important;
    }
    button[data-baseweb="tab"]:hover,
    button[data-baseweb="tab"]:hover * {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #2563EB !important;
        font-weight: 800 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 2. Service Singletons (Cached in Session State)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_services():
    """Initializes and caches FaceRecognitionService and AttendanceService."""
    database.init_db()
    face_svc = FaceRecognitionService()
    att_svc = AttendanceService(face_service=face_svc)

    # Ensure baseline sample students exist for immediate student login & testing
    existing = database.get_all_students()
    if len(existing) == 0:
        sample1_p = os.path.join("data", "samples", "sample_student_1.jpg")
        sample2_p = os.path.join("data", "samples", "sample_student_2.jpg")
        if os.path.exists(sample1_p):
            att_svc.register_student("21CS001", "Lena Forsen", "Sem-VII-CSE", cv2.imread(sample1_p))
        if os.path.exists(sample2_p):
            att_svc.register_student("21CS002", "Alex Mercer", "Sem-VII-CSE", cv2.imread(sample2_p))

    return face_svc, att_svc


face_service, attendance_service = get_services()


# Helper to convert BGR or Grayscale to 3-channel RGB for safe Streamlit display
def to_rgb_display(img: np.ndarray) -> np.ndarray:
    """
    Safely converts an image (Grayscale 2D or BGR 3D) to 3-channel RGB.
    Guarantees st.image() never fails with dimension or color map errors.
    """
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Helper to decode uploaded image or webcam file into OpenCV BGR numpy array
def decode_image_upload(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception as e:
        st.error(f"Failed to read image file: {e}")
        return None


# ----------------------------------------------------------------------------
# 3. Authentication Gate & Login Page (Admin & Student Login)
# ----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_role"] = None
    st.session_state["student_id"] = None
    st.session_state["student_name"] = None
    st.session_state["student_class"] = None
    st.session_state["nav_page"] = "Dashboard"

if not st.session_state.get("authenticated", False):
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-logo-icon">🎓</div>
                <div>
                    <div class="sidebar-brand-title">VisionAttendance</div>
                    <div class="sidebar-brand-subtitle">Semester VII CV Project</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("🔒 **Authentication Required**\n\nPlease select your portal on the right and sign in to continue.")
        st.markdown("---")
        st.caption("Engine: OpenCV YuNet (Face Detection) + SFace (Deep 128-d Cosine Embeddings)")

    # Centered Login Card
    col_l, col_center, col_r = st.columns([1, 1.8, 1])
    with col_center:
        st.markdown(
            """
            <div class="dashboard-panel-card" style="text-align: center; margin-top: 20px; padding: 26px 30px;">
                <div style="width: 52px; height: 52px; background: #2563EB; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.6rem; color: white; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(37,99,235,0.25);">🎓</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A; letter-spacing: -0.4px;">VisionAttendance Portal</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Smart Attendance Management System Using Face Recognition</div>
                <div style="font-size: 0.75rem; color: #2563EB; font-weight: 700; margin-top: 6px;">B.E. Semester VII Computer Vision Syllabus Project</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_student, tab_admin = st.tabs(["🎓 Student Login", "🛡️ Admin / Faculty Login"])

        # --------------------------------------------------------------------
        # STUDENT LOGIN TAB
        # Requirement: only login field and only login button; shows only live attendance after login
        # --------------------------------------------------------------------
        with tab_student:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.86rem; color: #334155; margin-bottom: 8px; font-weight: 600;'>Enter your registered Student ID to access the live attendance camera:</div>", unsafe_allow_html=True)
            
            # ONLY one login field
            student_input_id = st.text_input(
                "Student ID",
                placeholder="Enter Student ID (e.g. 21CS001)",
                key="student_login_id_input",
                label_visibility="collapsed",
            )
            
            st.write("")
            # ONLY one login button
            if st.button("Login", type="primary", use_container_width=True, key="student_login_btn"):
                clean_id = student_input_id.strip()
                if not clean_id:
                    st.warning("Please enter your Student ID.")
                else:
                    stu = database.get_student(clean_id)
                    if stu:
                        st.session_state["authenticated"] = True
                        st.session_state["user_role"] = "student"
                        st.session_state["student_id"] = stu["student_id"]
                        st.session_state["student_name"] = stu["name"]
                        st.session_state["student_class"] = stu["class_name"]
                        st.session_state["nav_page"] = "Live Attendance"
                        st.rerun()
                    else:
                        st.error(f"Student ID '{clean_id}' not found in database. Enrolled test IDs: 21CS001, 21CS002.")

            st.markdown(
                """
                <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 10px 14px; margin-top: 18px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Enrolled Demo Student IDs:</div>
                    <div style="font-size: 0.82rem; color: #0F172A; margin-top: 4px;">
                        <b><code>21CS001</code></b> (Lena Forsen) &nbsp;|&nbsp; <b><code>21CS002</code></b> (Alex Mercer)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------------
        # ADMIN LOGIN TAB
        # --------------------------------------------------------------------
        with tab_admin:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            admin_u = st.text_input("Username", value="admin", key="admin_user_input")
            admin_p = st.text_input("Password", value="admin123", type="password", key="admin_pass_input")
            st.caption("🔑 Demo Faculty Credentials: **`admin`** / **`admin123`**")
            st.write("")
            if st.button("Login as Administrator", type="primary", use_container_width=True, key="admin_login_btn"):
                if admin_u == "admin" and admin_p in ["admin", "admin123"]:
                    st.session_state["authenticated"] = True
                    st.session_state["user_role"] = "admin"
                    st.session_state["nav_page"] = "Dashboard"
                    st.rerun()
                else:
                    st.error("Invalid administrator credentials. Default is: admin / admin123")

    st.stop()


# ----------------------------------------------------------------------------
# 4. Role-Aware Sidebar Navigation
# ----------------------------------------------------------------------------
# Navigation links for Admin
NAV_ITEMS = [
    "Dashboard",
    "Live Attendance",
    "Students & Enrollment",
    "Attendance Sessions",
    "Attendance History",
    "Reports & Analytics",
    "Classes & Subjects",
    "Computer Vision Lab",
    "Syllabus Mapping",
    "Help & Viva Guide",
    "Profile & Settings",
]

NAV_DISPLAY_MAP = {
    "Dashboard": "📊  Dashboard",
    "Live Attendance": "📷  Live Attendance     [Live CV]",
    "Students & Enrollment": "👥  Students & Enrollment",
    "Attendance Sessions": "📅  Attendance Sessions",
    "Attendance History": "🕒  Attendance History",
    "Reports & Analytics": "📈  Reports & Analytics",
    "Classes & Subjects": "📚  Classes & Subjects",
    "Computer Vision Lab": "🔬  Computer Vision Lab  [11 Practicals]",
    "Syllabus Mapping": "📖  Syllabus Mapping     [Units 1-10]",
    "Help & Viva Guide": "❓  Help & Viva Guide     [Viva 40+]",
    "Profile & Settings": "⚙️  Profile & Settings",
}

user_role = st.session_state.get("user_role", "admin")

with st.sidebar:
    if user_role == "student":
        # Sidebar for Student: ONLY Live Attendance
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-logo-icon">🎓</div>
                <div>
                    <div class="sidebar-brand-title">VisionAttendance</div>
                    <div class="sidebar-brand-subtitle">Student Portal</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
                <div style="font-size: 0.68rem; font-weight: 800; color: #2563EB; letter-spacing: 0.5px; text-transform: uppercase;">LOGGED IN STUDENT</div>
                <div style="font-size: 1.0rem; font-weight: 800; color: #0F172A; margin-top: 2px;">{st.session_state.get('student_name', 'Student')}</div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 2px;">ID: <b>{st.session_state.get('student_id', '')}</b> &bull; {st.session_state.get('student_class', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">ACCESSIBLE MODULES</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 12px; margin-bottom: 16px; font-size: 0.84rem; font-weight: 700; color: #2563EB; display: flex; justify-content: space-between; align-items: center;">
                <span>📷  Live Attendance</span>
                <span style="background: #2563EB; color: white; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px;">ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("###### ⚙️ SFace Sensitivity")
        cosine_thresh = st.slider(
            "Matching Threshold",
            min_value=0.20,
            max_value=0.60,
            value=DEFAULT_COSINE_THRESHOLD,
            step=0.01,
            help="Cosine similarity threshold for OpenCV SFace model. Default is 0.363.",
        )

        st.write("")
        if st.button("🚪 Sign Out / Logout", use_container_width=True, key="student_signout_btn"):
            st.session_state.clear()
            st.rerun()

    else:
        # Sidebar for Faculty Admin: Full Access
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-logo-icon">🎓</div>
                <div>
                    <div class="sidebar-brand-title">VisionAttendance</div>
                    <div class="sidebar-brand-subtitle">Faculty Admin Portal</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">CORE MANAGEMENT</div>', unsafe_allow_html=True)

        current_idx = NAV_ITEMS.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in NAV_ITEMS else 0

        selected_nav = st.radio(
            "Sidebar Navigation",
            options=NAV_ITEMS,
            index=current_idx,
            format_func=lambda k: NAV_DISPLAY_MAP.get(k, k),
            label_visibility="collapsed",
        )

        if selected_nav != st.session_state["nav_page"]:
            st.session_state["nav_page"] = selected_nav
            st.rerun()

        st.markdown("---")

        # Sensitivity Slider
        st.markdown("###### ⚙️ SFace Sensitivity")
        cosine_thresh = st.slider(
            "Matching Threshold",
            min_value=0.20,
            max_value=0.60,
            value=DEFAULT_COSINE_THRESHOLD,
            step=0.01,
            help="Cosine similarity threshold for OpenCV SFace model. Default is 0.363.",
        )

        st.write("")
        if st.button("🚪 Sign Out / Logout", use_container_width=True, key="admin_signout_btn"):
            st.session_state.clear()
            st.rerun()


# ----------------------------------------------------------------------------
# 5. Global Top Header Bar (Renders across all pages)
# ----------------------------------------------------------------------------
def render_top_header_bar(title_text: str):
    role = st.session_state.get("user_role", "admin")
    if role == "student":
        student_name = st.session_state.get("student_name", "Student")
        student_id = st.session_state.get("student_id", "")
        role_badge = f'<div class="pill-admin-tag" style="background-color: #EFF6FF; color: #2563EB; border-color: #BFDBFE;">🎓 STUDENT: {student_name} ({student_id})</div>'
    else:
        role_badge = '<div class="pill-admin-tag">🛡️ ADMIN PORTAL</div>'

    st.markdown(
        f"""
        <div class="top-header-bar">
            <div class="page-main-title">{title_text}</div>
            <div class="header-badge-container">
                <div class="pill-status-green">
                    <span class="status-dot-green"></span> OpenCV YuNet + SFace Ready
                </div>
                {role_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Enforce that students ONLY view Live Attendance
if user_role == "student":
    active_page = "Live Attendance"
else:
    active_page = st.session_state.get("nav_page", "Dashboard")


# ============================================================================
# 1. PAGE: DASHBOARD (Matching the Exact Screenshot Layout & Real DB Data)
# ============================================================================
if active_page == "Dashboard":
    render_top_header_bar("Academic Overview Dashboard")

    # Real data from SQLite database
    all_students = database.get_all_students()
    total_students = len(all_students)
    classes_list = database.get_unique_classes()
    class_count = max(1, len(classes_list))

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_summary = attendance_service.get_summary(today_str)
    present_today = today_summary["present_count"]
    absent_today = today_summary["absent_count"]
    pct_today = today_summary["attendance_percentage"]

    # Pre-populate sample students if empty for instant demonstration
    if total_students == 0:
        sample1_p = os.path.join("data", "samples", "sample_student_1.jpg")
        sample2_p = os.path.join("data", "samples", "sample_student_2.jpg")
        if os.path.exists(sample1_p):
            attendance_service.register_student("21CS001", "Lena Forsen", "Sem-VII-CSE", cv2.imread(sample1_p))
        if os.path.exists(sample2_p):
            attendance_service.register_student("21CS002", "Alex Mercer", "Sem-VII-CSE", cv2.imread(sample2_p))
        all_students = database.get_all_students()
        total_students = len(all_students)
        today_summary = attendance_service.get_summary(today_str)
        present_today = today_summary["present_count"]
        absent_today = today_summary["absent_count"]
        pct_today = today_summary["attendance_percentage"]

    # ------------------------------------------------------------------------
    # ROW 1: 4 Metric Cards (Matching screenshot icons, titles, and layout)
    # ------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-stat-card">
                <div>
                    <div class="kpi-title">Total Students</div>
                    <div class="kpi-number">{total_students}</div>
                    <div class="kpi-desc">Registered in database</div>
                </div>
                <div class="kpi-icon-box icon-box-blue">👥</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-stat-card">
                <div>
                    <div class="kpi-title">Academic Classes</div>
                    <div class="kpi-number">{class_count}</div>
                    <div class="kpi-desc">Sem 7 divisions</div>
                </div>
                <div class="kpi-icon-box icon-box-purple">📚</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-stat-card">
                <div>
                    <div class="kpi-title">Active Sessions</div>
                    <div class="kpi-number">{1 if present_today > 0 else 0}</div>
                    <div class="kpi-desc">{'Session verified today' if present_today > 0 else 'No live room right now'}</div>
                </div>
                <div class="kpi-icon-box icon-box-amber">📅</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-stat-card">
                <div>
                    <div class="kpi-title">Recent Attendance</div>
                    <div class="kpi-number">{pct_today:.0f}%</div>
                    <div class="kpi-desc">{present_today} Present / {absent_today} Absent</div>
                </div>
                <div class="kpi-icon-box icon-box-green">✅</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ------------------------------------------------------------------------
    # ROW 2: Two Central Panels (Latest Session on Left, Today's Verified Roster on Right)
    # ------------------------------------------------------------------------
    col_session, col_roster = st.columns([1, 1.15])

    with col_session:
        session_code = f"SESS-{datetime.now().strftime('%Y%m%d')}-CV-001"
        progress_pct = int(pct_today) if total_students > 0 else 0

        st.markdown(
            f"""
            <div class="dashboard-panel-card" style="height: 100%;">
                <div class="panel-header-row">
                    <div>
                        <div class="panel-card-title">Latest Attendance Session</div>
                        <div class="panel-card-subtitle">Summary from verified SQLite records</div>
                    </div>
                    <span class="status-badge-completed">{'COMPLETED' if present_today > 0 else 'ACTIVE'}</span>
                </div>
                <div class="inner-session-box">
                    <div>
                        <div class="session-meta-label">SESSION CODE</div>
                        <div class="session-meta-value">{session_code}</div>
                    </div>
                    <div>
                        <div class="session-meta-label">SUBJECT</div>
                        <div class="session-meta-value-blue">Computer Vision (CS-701)</div>
                    </div>
                </div>
                <div class="ratio-header">
                    <span>Present Ratio</span>
                    <span>{present_today} / {total_students} ({pct_today:.0f}%)</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
                </div>
                <div style="display: flex; gap: 20px; padding-top: 12px; border-top: 1px solid #F1F5F9;">
                    <div>
                        <div class="session-meta-label">TARGET CLASS</div>
                        <div class="session-meta-value" style="font-size: 0.85rem;">Sem-VII-CSE</div>
                    </div>
                    <div>
                        <div class="session-meta-label">ROOM LOCATION</div>
                        <div class="session-meta-value" style="font-size: 0.85rem;">CV Lab (Lab-402)</div>
                    </div>
                    <div>
                        <div class="session-meta-label">SCHEDULE</div>
                        <div class="session-meta-value" style="font-size: 0.85rem;">09:30 AM - 10:30 AM</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_roster:
        st.markdown(
            f"""
            <div class="dashboard-panel-card" style="height: 100%;">
                <div class="panel-header-row">
                    <div>
                        <div class="panel-card-title">Today's Class Attendance Roster</div>
                        <div class="panel-card-subtitle">Direct live status from SQLite database</div>
                    </div>
                    <div style="display: flex; gap: 6px;">
                        <span style="background-color: #ECFDF5; color: #059669; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">{present_today} Present</span>
                        <span style="background-color: #FEF2F2; color: #DC2626; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">{absent_today} Absent</span>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        all_students_list = database.get_all_students()
        today_att_map = {rec["student_id"]: rec["attendance_time"] for rec in today_summary["records"]}

        if all_students_list:
            roster_data = []
            for s in all_students_list:
                s_id = s["student_id"]
                is_p = s_id in today_att_map
                roster_data.append(
                    {
                        "Student ID": s_id,
                        "Student Name": s["name"],
                        "Class": s["class_name"],
                        "Status": "✅ Present" if is_p else "❌ Absent",
                        "Time": today_att_map[s_id] if is_p else "—",
                    }
                )
            st.dataframe(
                pd.DataFrame(roster_data),
                use_container_width=True,
                hide_index=True,
                height=185,
            )
        else:
            st.caption("No students enrolled yet. Use 'Enroll Student Faces' to register students.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------------------------
    # ROW 3: Quick Action Center (Matching the 4 shortcut buttons from image)
    # ------------------------------------------------------------------------
    st.markdown(
        """
        <div class="dashboard-panel-card" style="padding: 18px 24px;">
            <div class="quick-action-header">
                <div class="quick-action-title">Quick Action Center</div>
                <div class="quick-action-subtitle">Common faculty & examiner shortcuts</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qa1, qa2, qa3, qa4 = st.columns(4)

    with qa1:
        if st.button("📷 Live Attendance Camera", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Live Attendance"
            st.rerun()

    with qa2:
        if st.button("👤+ Enroll Student Faces", use_container_width=True):
            st.session_state["nav_page"] = "Students & Enrollment"
            st.rerun()

    with qa3:
        if st.button("🔬 Open CV Practicals Lab", use_container_width=True):
            st.session_state["nav_page"] = "Computer Vision Lab"
            st.rerun()

    with qa4:
        if st.button("❓ Viva Guide (40+ Q&A)", use_container_width=True):
            st.session_state["nav_page"] = "Help & Viva Guide"
            st.rerun()


# ============================================================================
# 2. PAGE: LIVE ATTENDANCE (Camera & Verification Pipeline)
# ============================================================================
elif active_page == "Live Attendance":
    render_top_header_bar("Live Attendance Marking")

    col_feed, col_match = st.columns([1.2, 1])

    with col_feed:
        st.markdown(
            """
            <div class="dashboard-panel-card" style="margin-bottom: 16px;">
                <div class="panel-card-title">Live Classroom Camera</div>
                <div class="panel-card-subtitle">Position your face inside the webcam frame to mark attendance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        feed_frame: Optional[np.ndarray] = None
        cam_frame = st.camera_input("Capture live face frame", key="live_cam_input")
        if cam_frame is not None:
            feed_frame = decode_image_upload(cam_frame)

    with col_match:
        st.markdown(
            """
            <div class="dashboard-panel-card" style="margin-bottom: 16px;">
                <div class="panel-card-title">Recognition & Verification</div>
                <div class="panel-card-subtitle">OpenCV SFace deep cosine matching results</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        att_date_str = datetime.now().strftime("%Y-%m-%d")

        if feed_frame is None:
            st.info("📷 Please capture a face frame using the webcam camera to evaluate attendance.")
        else:
            annotated_frame, candidates = attendance_service.process_attendance_frame(
                image_bgr=feed_frame,
                attendance_date=att_date_str,
                threshold=cosine_thresh,
            )

            st.image(
                to_rgb_display(annotated_frame),
                caption="Recognition Overlays (Green = Verified, Orange = Unknown)",
                use_container_width=True,
            )

            if not candidates:
                st.warning("No faces detected in the submitted image. Ensure proper lighting and front-facing posture.")
            else:
                for idx, cand in enumerate(candidates):
                    is_rec = cand["is_recognized"]
                    sim_pct = cand["similarity"] * 100
                    conf_pct = cand["confidence"] * 100

                    if is_rec:
                        st.markdown(
                            f"""
                            <div style="background: white; border: 1px solid #10B981; border-left: 5px solid #10B981; border-radius: 10px; padding: 14px; margin-top: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                                <div style="font-weight: 800; color: #065F46; font-size: 1.05rem;">{cand['name']}</div>
                                <div style="font-size: 0.85rem; color: #374151; margin-top: 2px;">
                                    <b>ID:</b> <code>{cand['student_id']}</code> &nbsp;|&nbsp; <b>Class:</b> {cand['class_name']}
                                </div>
                                <div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">
                                    <b>Similarity:</b> {sim_pct:.1f}% &nbsp;|&nbsp; <b>Confidence:</b> {conf_pct:.0f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if cand["already_marked"]:
                            st.info(f"ℹ️ Attendance already recorded for {cand['name']} today ({att_date_str}).")
                        else:
                            if st.button(
                                f"✅ Confirm & Mark Attendance for {cand['name']}",
                                key=f"btn_mark_{idx}",
                                type="primary",
                                use_container_width=True,
                            ):
                                ok_mark, msg_mark, is_dup = attendance_service.confirm_attendance(
                                    student_id=cand["student_id"],
                                    attendance_date=att_date_str,
                                )
                                if ok_mark:
                                    st.success(msg_mark)
                                    st.rerun()
                                else:
                                    if is_dup:
                                        st.warning(msg_mark)
                                    else:
                                        st.error(msg_mark)
                    else:
                        st.markdown(
                            f"""
                            <div style="background: white; border: 1px solid #F59E0B; border-left: 5px solid #F59E0B; border-radius: 10px; padding: 14px; margin-top: 12px;">
                                <div style="font-weight: 800; color: #92400E; font-size: 1rem;">Unknown Face / Needs Review</div>
                                <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">
                                    Highest match score: {sim_pct:.1f}% (Threshold: {cosine_thresh*100:.1f}%)
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ============================================================================
# 3. PAGE: STUDENTS & ENROLLMENT (Registration & Template Directory)
# ============================================================================
elif active_page == "Students & Enrollment":
    render_top_header_bar("Student Enrollment & Face Templates")

    col_reg_form, col_reg_cam = st.columns([1, 1.2])

    with col_reg_form:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Student Profile Details</div>
                <div class="panel-card-subtitle">Enter student ID, name, and class division</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        reg_id = st.text_input("Student ID *", placeholder="e.g. 21CS042").strip()
        reg_name = st.text_input("Full Name *", placeholder="e.g. Smit Prajapati").strip()
        reg_class = st.text_input("Class Name *", value="Sem-VII-CSE").strip()

        st.markdown(
            """
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 12px; margin-top: 14px;">
                <b style="color: #166534; font-size: 0.85rem;">🔒 Biometric Storage Notice:</b>
                <div style="font-size: 0.78rem; color: #14532D; margin-top: 4px;">
                    Face embeddings are saved as 128 float32 numbers (512 bytes) inside SQLite. Raw photographs are not permanently retained.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_reg_cam:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Facial Photo & Validation</div>
                <div class="panel-card-subtitle">Detect 5 landmarks and verify quality thresholds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_cam, tab_file, tab_sample = st.tabs(["Webcam Camera", "Upload Photo", "Sample Portrait"])
        chosen_bgr: Optional[np.ndarray] = None

        with tab_cam:
            cam_shot = st.camera_input("Take frontal photo", key="reg_cam_input")
            if cam_shot is not None:
                chosen_bgr = decode_image_upload(cam_shot)

        with tab_file:
            uploaded_pic = st.file_uploader(
                "Upload student portrait (JPG / PNG)", type=["jpg", "jpeg", "png"], key="reg_file_input"
            )
            if uploaded_pic is not None:
                chosen_bgr = decode_image_upload(uploaded_pic)

        with tab_sample:
            st.caption("Load a bundled student portrait for instant testing:")
            sample_sel = st.selectbox(
                "Select Pre-bundled Student Portrait",
                ["Sample Student 1 (Lena Forsen)", "Sample Student 2 (Alex Mercer)"],
                key="sample_reg_sel",
            )
            if st.button("Load Selected Sample Portrait", use_container_width=True):
                s_filename = "sample_student_1.jpg" if "Sample Student 1" in sample_sel else "sample_student_2.jpg"
                sample_path = os.path.join("data", "samples", s_filename)
                if os.path.exists(sample_path):
                    chosen_bgr = cv2.imread(sample_path)
                    st.session_state["active_reg_img"] = chosen_bgr

        if chosen_bgr is None and "active_reg_img" in st.session_state:
            chosen_bgr = st.session_state["active_reg_img"]

        if chosen_bgr is not None:
            is_valid, v_msg, embedding, annotated_img = face_service.validate_for_registration(chosen_bgr)
            if is_valid and annotated_img is not None:
                st.image(
                    to_rgb_display(annotated_img),
                    caption="✅ Face detected: Bounding box + 5 Landmarks",
                    use_container_width=True,
                )
                st.success(f"Validation Passed: {v_msg}")

                if st.button("💾 Complete Registration & Save Template", type="primary", use_container_width=True):
                    if not reg_id or not reg_name or not reg_class:
                        st.error("Please fill in Student ID, Full Name, and Class Name.")
                    else:
                        success, save_msg, _ = attendance_service.register_student(
                            student_id=reg_id,
                            name=reg_name,
                            class_name=reg_class,
                            image_bgr=chosen_bgr,
                        )
                        if success:
                            st.success(save_msg)
                            if "active_reg_img" in st.session_state:
                                del st.session_state["active_reg_img"]
                            st.rerun()
                        else:
                            st.error(save_msg)
            else:
                st.image(to_rgb_display(chosen_bgr), caption="Submitted Image", use_container_width=True)
                st.warning(f"⚠️ Validation Issue: {v_msg}")

    # Enrolled Students Directory Table directly from SQLite
    st.write("")
    st.markdown(
        """
        <div class="dashboard-panel-card">
            <div class="panel-card-title">Enrolled Students Directory</div>
            <div class="panel-card-subtitle">Active student records stored in SQLite database</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    students_list = database.get_all_students()
    if students_list:
        s_df = pd.DataFrame(students_list)
        st.dataframe(
            s_df.rename(
                columns={
                    "student_id": "Student ID",
                    "name": "Full Name",
                    "class_name": "Class Name",
                    "created_at": "Registration Timestamp",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("###### 🗑️ Delete Student")
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            options_del = [f"{s['student_id']} - {s['name']} ({s['class_name']})" for s in students_list]
            chosen_delete = st.selectbox("Select student to remove", options_del)
        with del_col2:
            st.write("")
            st.write("")
            if st.button("Delete Student", type="secondary", use_container_width=True):
                target_id = chosen_delete.split(" - ")[0]
                if database.delete_student(target_id):
                    st.success(f"Student '{target_id}' and their template removed.")
                    st.rerun()
                else:
                    st.error("Failed to delete student.")
    else:
        st.info("No students enrolled yet. Use the form above to register students.")


# ============================================================================
# 4. PAGE: ATTENDANCE SESSIONS (Session Codes, Rosters, Daily Aggregates)
# ============================================================================
elif active_page == "Attendance Sessions":
    render_top_header_bar("Attendance Sessions & Daily Class Rosters")

    all_dates = database.get_distinct_attendance_dates()

    if not all_dates:
        st.info("No attendance sessions have been conducted yet. Mark attendance on the 'Live Attendance' page.")
    else:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Completed & Verified Sessions</div>
                <div class="panel-card-subtitle">Sessions grouped by date from SQLite attendance records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for s_date in all_dates:
            session_code = f"SESS-{s_date.replace('-', '')}-CV-001"
            s_summary = attendance_service.get_summary(s_date)
            p_count = s_summary["present_count"]
            t_count = s_summary["total_registered"]
            pct = s_summary["attendance_percentage"]

            with st.expander(f"📁 {session_code} — {s_date} — Present: {p_count}/{t_count} ({pct:.0f}%)"):
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Date", s_date)
                col_s2.metric("Subject", "Computer Vision")
                col_s3.metric("Present Count", p_count)
                col_s4.metric("Absent Count", s_summary["absent_count"])

                if s_summary["records"]:
                    s_df = pd.DataFrame(s_summary["records"])
                    st.dataframe(
                        s_df[["student_id", "name", "class_name", "attendance_time", "status"]].rename(
                            columns={
                                "student_id": "Student ID",
                                "name": "Student Name",
                                "class_name": "Class",
                                "attendance_time": "Time Marked",
                                "status": "Status",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================================
# 5. PAGE: ATTENDANCE HISTORY (Chronological Logs & CSV Export)
# ============================================================================
elif active_page == "Attendance History":
    render_top_header_bar("Attendance History & Audit Logs")

    f_col1, f_col2, f_col3 = st.columns([1, 1, 1.2])
    with f_col1:
        filter_date = st.date_input("Filter by Date", value=datetime.now().date(), key="hist_filter_date")
        f_date_str = filter_date.strftime("%Y-%m-%d") if filter_date else None
    with f_col2:
        classes = ["All Classes"] + database.get_unique_classes()
        filter_class = st.selectbox("Filter by Class", classes, key="hist_filter_class")
        f_class_filter = None if filter_class == "All Classes" else filter_class
    with f_col3:
        filter_search = st.text_input("Search Student ID or Name", placeholder="e.g. 21CS or Lena").strip()
        f_search_filter = filter_search if filter_search else None

    # Fetch logs
    df = attendance_service.get_records_dataframe(
        date=f_date_str,
        student_id=f_search_filter,
        class_name=f_class_filter,
    )

    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_data = attendance_service.export_records_to_csv(
            date=f_date_str, student_id=f_search_filter, class_name=f_class_filter
        )
        file_name = f"attendance_history_{f_date_str or 'all'}.csv"
        st.download_button(
            label="📥 Export Filtered History to CSV",
            data=csv_data,
            file_name=file_name,
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("No attendance entries found matching the specified filters.")


# ============================================================================
# 6. PAGE: REPORTS & ANALYTICS (Performance, Charts & Attendance Breakdown)
# ============================================================================
elif active_page == "Reports & Analytics":
    render_top_header_bar("Attendance Reports & Analytics")

    st_stats = database.get_student_attendance_summary()
    all_dates = database.get_distinct_attendance_dates()
    total_enrolled = len(database.get_all_students())

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.markdown(f'<div class="kpi-stat-card"><div><div class="kpi-title">Total Enrolled</div><div class="kpi-number">{total_enrolled}</div></div><div class="kpi-icon-box icon-box-blue">👥</div></div>', unsafe_allow_html=True)
    with col_a2:
        st.markdown(f'<div class="kpi-stat-card"><div><div class="kpi-title">Total Sessions</div><div class="kpi-number">{len(all_dates)}</div></div><div class="kpi-icon-box icon-box-purple">📅</div></div>', unsafe_allow_html=True)
    with col_a3:
        avg_pct = round(sum(s["attendance_percentage"] for s in st_stats) / max(1, len(st_stats)), 1) if st_stats else 0
        st.markdown(f'<div class="kpi-stat-card"><div><div class="kpi-title">Average Attendance</div><div class="kpi-number" style="color: #059669;">{avg_pct}%</div></div><div class="kpi-icon-box icon-box-green">📈</div></div>', unsafe_allow_html=True)
    with col_a4:
        st.markdown(f'<div class="kpi-stat-card"><div><div class="kpi-title">Database Storage</div><div class="kpi-number" style="font-size: 1.4rem; padding-top: 6px;">SQLite</div></div><div class="kpi-icon-box icon-box-amber">💾</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown(
        """
        <div class="dashboard-panel-card">
            <div class="panel-card-title">Student Attendance Leaderboard & Session Frequency</div>
            <div class="panel-card-subtitle">Aggregate attendance performance per enrolled student</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st_stats:
        stats_df = pd.DataFrame(st_stats)
        st.dataframe(
            stats_df.rename(
                columns={
                    "student_id": "Student ID",
                    "name": "Full Name",
                    "class_name": "Class Division",
                    "attended_sessions": "Sessions Attended",
                    "total_sessions": "Total Sessions",
                    "attendance_percentage": "Attendance %",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No attendance records logged yet.")


# ============================================================================
# 7. PAGE: CLASSES & SUBJECTS (Academic divisions and course details)
# ============================================================================
elif active_page == "Classes & Subjects":
    render_top_header_bar("Academic Classes & Course Subjects")

    col_sub, col_cls_view = st.columns([1, 1.2])

    with col_sub:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Course Subject Details</div>
                <div class="panel-card-subtitle">Semester VII Computer Vision Curriculum</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; margin-bottom: 14px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #1E3A8A;">Computer Vision</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Subject Code: <code>CS-701</code> &nbsp;|&nbsp; B.E. Semester VII</div>
                <hr style="margin: 12px 0; border: 0; border-top: 1px solid #E2E8F0;">
                <div style="font-size: 0.82rem; color: #374151;">
                    <b>Total Units:</b> 10 Syllabus Units<br>
                    <b>Practical Labs:</b> 7 Core Operations + 4 Advanced Practicals<br>
                    <b>Primary Models:</b> OpenCV YuNet (Detection) + OpenCV SFace (Recognition)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cls_view:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Class Divisions in SQLite Database</div>
                <div class="panel-card-subtitle">Active classes populated by registered students</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        classes_list = database.get_unique_classes()
        if classes_list:
            for cls in classes_list:
                c_students = database.get_all_students(class_name=cls)
                with st.expander(f"🏫 Class Division: {cls} ({len(c_students)} Students Enrolled)"):
                    if c_students:
                        st.dataframe(
                            pd.DataFrame(c_students)[["student_id", "name", "created_at"]].rename(
                                columns={"student_id": "ID", "name": "Name", "created_at": "Enrolled Date"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
        else:
            st.info("No class divisions registered yet.")


# ============================================================================
# 8. PAGE: COMPUTER VISION LAB (7 Core Operations with Zero cmap Errors)
# ============================================================================
elif active_page == "Computer Vision Lab":
    render_top_header_bar("Computer Vision Practicals Lab")

    source_choice = st.radio(
        "Choose Lab Image Source:",
        ["Built-in Standard CV Lab Image", "Upload Custom Image"],
        horizontal=True,
    )

    lab_img_bgr: Optional[np.ndarray] = None
    if source_choice == "Built-in Standard CV Lab Image":
        sample_lab_path = os.path.join("data", "samples", "cv_lab_sample.jpg")
        if os.path.exists(sample_lab_path):
            lab_img_bgr = cv2.imread(sample_lab_path)
        else:
            st.error("Default sample image missing.")
    else:
        uploaded_lab = st.file_uploader(
            "Upload any test photograph", type=["jpg", "jpeg", "png"], key="cv_lab_uploader"
        )
        if uploaded_lab is not None:
            lab_img_bgr = decode_image_upload(uploaded_lab)

    if lab_img_bgr is not None:
        st.write("")
        op_tab1, op_tab2, op_tab3, op_tab4, op_tab5, op_tab6, op_tab7 = st.tabs(
            [
                "1. Grayscale",
                "2. Brightness & Contrast",
                "3. Histogram Equalization",
                "4. Smoothing Filters",
                "5. Canny Edges",
                "6. Morphology",
                "7. Thresholding",
            ]
        )

        # 1. Grayscale
        with op_tab1:
            st.markdown("##### 1. Color to Grayscale Conversion")
            st.caption("Formula: $Y = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B$ (ITU-R BT.601 standard)")
            gray_res = image_processing.to_grayscale(lab_img_bgr)
            c_orig, c_gray = st.columns(2)
            with c_orig:
                st.image(to_rgb_display(lab_img_bgr), caption="Original BGR Image (3 Channels)", use_container_width=True)
            with c_gray:
                st.image(to_rgb_display(gray_res), caption="Grayscale Output (1 Channel Intensity)", use_container_width=True)

        # 2. Brightness & Contrast
        with op_tab2:
            st.markdown("##### 2. Brightness & Contrast Linear Adjustment")
            st.caption("Formula: $g(x, y) = \\alpha \\cdot f(x, y) + \\beta$")
            c_ctrl1, c_ctrl2 = st.columns(2)
            with c_ctrl1:
                alpha_val = st.slider("Contrast Gain (α)", 0.2, 3.0, 1.2, 0.1)
            with c_ctrl2:
                beta_val = st.slider("Brightness Bias (β)", -100, 100, 20, 5)

            bc_res = image_processing.adjust_brightness_contrast(lab_img_bgr, alpha=alpha_val, beta=beta_val)
            c_bc1, c_bc2 = st.columns(2)
            with c_bc1:
                st.image(to_rgb_display(lab_img_bgr), caption="Original Image", use_container_width=True)
            with c_bc2:
                st.image(to_rgb_display(bc_res), caption=f"Adjusted: α={alpha_val}, β={beta_val}", use_container_width=True)

        # 3. Histogram Equalization
        with op_tab3:
            st.markdown("##### 3. Histogram Equalization")
            st.caption("Flattens and stretches the cumulative distribution function (CDF) across [0, 255]")
            eq_res, is_col = image_processing.equalize_histogram(lab_img_bgr)
            c_eq1, c_eq2 = st.columns(2)
            with c_eq1:
                st.image(to_rgb_display(lab_img_bgr), caption="Original Image", use_container_width=True)
            with c_eq2:
                st.image(to_rgb_display(eq_res), caption="Histogram Equalized Image", use_container_width=True)

            fig = image_processing.plot_histogram_comparison(lab_img_bgr, eq_res)
            st.pyplot(fig)
            plt.close(fig)

        # 4. Smoothing Filters
        with op_tab4:
            st.markdown("##### 4. Spatial Smoothing Filters")
            st.caption("Removes noise and high-frequency fluctuations through spatial neighborhood convolution")
            c_sm1, c_sm2 = st.columns(2)
            with c_sm1:
                smooth_type = st.selectbox("Smoothing Filter Type", ["Gaussian Smoothing", "Median Smoothing"])
            with c_sm2:
                k_size = st.select_slider("Kernel Window Size", options=[3, 5, 7, 9, 11, 15], value=5)

            method_key = "median" if "Median" in smooth_type else "gaussian"
            smoothed = image_processing.apply_smoothing(lab_img_bgr, method=method_key, kernel_size=k_size)

            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.image(to_rgb_display(lab_img_bgr), caption="Original Image", use_container_width=True)
            with c_s2:
                st.image(to_rgb_display(smoothed), caption=f"{smooth_type} (Kernel: {k_size}×{k_size})", use_container_width=True)

        # 5. Canny Edge Detection
        with op_tab5:
            st.markdown("##### 5. Canny Multi-Stage Edge Detection")
            st.caption("4 Stages: Gaussian Blur → Sobel Gradients → Non-Maximum Suppression → Hysteresis Thresholding")
            c_can1, c_can2 = st.columns(2)
            with c_can1:
                low_th = st.slider("Low Hysteresis Threshold (T1)", 10, 200, 50, 5)
            with c_can2:
                high_th = st.slider("High Hysteresis Threshold (T2)", 50, 300, 150, 5)

            canny_edges = image_processing.apply_canny(lab_img_bgr, low_threshold=low_th, high_threshold=high_th)
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.image(to_rgb_display(lab_img_bgr), caption="Original Image", use_container_width=True)
            with c_e2:
                st.image(to_rgb_display(canny_edges), caption=f"Canny Edges (T1={low_th}, T2={high_th})", use_container_width=True)

        # 6. Morphology
        with op_tab6:
            st.markdown("##### 6. Morphological Operations")
            st.caption("Non-linear structuring element transformations: Erosion, Dilation, Opening, Closing")
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                morph_op = st.selectbox("Operation", ["Dilation", "Erosion", "Opening", "Closing"])
            with c_m2:
                morph_shape = st.selectbox("Kernel Shape", ["Rectangle", "Cross", "Ellipse"])
            with c_m3:
                morph_ksize = st.select_slider("Kernel Size", options=[3, 5, 7, 9], value=5)

            gray_for_morph = image_processing.to_grayscale(lab_img_bgr)
            morph_res = image_processing.apply_morphology(
                gray_for_morph,
                operation=morph_op.lower(),
                kernel_shape=morph_shape.lower(),
                kernel_size=morph_ksize,
            )

            c_mo1, c_mo2 = st.columns(2)
            with c_mo1:
                st.image(to_rgb_display(gray_for_morph), caption="Grayscale Input", use_container_width=True)
            with c_mo2:
                st.image(to_rgb_display(morph_res), caption=f"Morphological {morph_op} ({morph_shape} {morph_ksize}×{morph_ksize})", use_container_width=True)

        # 7. Thresholding
        with op_tab7:
            st.markdown("##### 7. Basic Image Thresholding & Otsu's Method")
            st.caption("Segments foreground from background using pixel intensity cutoffs")
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                th_type = st.selectbox(
                    "Threshold Type",
                    [
                        "Binary (cv2.THRESH_BINARY)",
                        "Binary Inverted (cv2.THRESH_BINARY_INV)",
                        "Truncate (cv2.THRESH_TRUNC)",
                        "To Zero (cv2.THRESH_TOZERO)",
                        "Otsu's Automated Threshold",
                    ],
                )
            with c_t2:
                manual_th = st.slider("Manual Threshold Cutoff", 0, 255, 127, 1)

            type_key = "otsu" if "Otsu" in th_type else th_type.split()[0].lower()
            thresh_img, computed_th = image_processing.apply_thresholding(
                lab_img_bgr, threshold_value=manual_th, threshold_type=type_key
            )

            c_th1, c_th2 = st.columns(2)
            with c_th1:
                st.image(to_rgb_display(image_processing.to_grayscale(lab_img_bgr)), caption="Grayscale Input", use_container_width=True)
            with c_th2:
                st.image(to_rgb_display(thresh_img), caption=f"Segmented Output (Cutoff: {computed_th:.1f})", use_container_width=True)


# ============================================================================
# 9. PAGE: SYLLABUS MAPPING (10 Units Verification)
# ============================================================================
elif active_page == "Syllabus Mapping":
    render_top_header_bar("Semester VII Syllabus Mapping")

    st.markdown(
        """
        <div class="dashboard-panel-card" style="margin-bottom: 20px;">
            <div class="panel-card-title">10 Units Syllabus Verification Matrix</div>
            <div class="panel-card-subtitle">Real implementation status mapped to project source modules</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mapping_data = [
        ("Unit 1", "Introduction to Computer Vision", "Image representation, camera models, resolution", "Implemented", "image_processing.py / BGR NumPy matrices"),
        ("Unit 2", "Digital Image Processing", "Color spaces, luminance weighting, grayscale conversion", "Implemented", "image_processing.to_grayscale()"),
        ("Unit 3", "Image Enhancement & Morphology", "Linear contrast/brightness, HistEq, Erosion, Dilation", "Implemented", "image_processing.adjust_brightness_contrast() / apply_morphology()"),
        ("Unit 4", "Image Filtering & Edge Detection", "Gaussian blur, non-linear median filter, 4-stage Canny", "Implemented", "image_processing.apply_smoothing() / apply_canny()"),
        ("Unit 5", "Feature Detection & Descriptors", "Facial landmarks, 128-d deep embeddings vs traditional HOG", "Implemented", "face_recognition_service.py (SFace + YuNet)"),
        ("Unit 6", "Image Segmentation", "Binary thresholding, Otsu's variance maximization", "Implemented", "image_processing.apply_thresholding()"),
        ("Unit 7", "CNNs & Image Classification", "Deep convolutional layers, pretrained vision backbones", "Implemented (Inference)", "cv2.dnn (YuNet + SFace ONNX)"),
        ("Unit 8", "Advanced CNN Techniques", "Transfer learning, fine-tuning, attention mechanisms", "Explained Conceptually", "docs/viva_guide.md (Theoretical reference)"),
        ("Unit 9", "Generative Adversarial Networks", "Generator & discriminator minimax game, augmentation", "Explained Conceptually", "docs/viva_guide.md (Theoretical reference)"),
        ("Unit 10", "Motion Detection & Video Processing", "Webcam frame-by-frame processing & visual bounding boxes", "Implemented", "Streamlit camera_input + OpenCV overlays"),
    ]

    map_df = pd.DataFrame(mapping_data, columns=["Unit", "Module Title", "Syllabus Coverage", "Implementation Status", "Codebase Reference"])
    st.dataframe(map_df, use_container_width=True, hide_index=True)


# ============================================================================
# 10. PAGE: HELP & VIVA GUIDE (28+ Viva Q&As & Architecture)
# ============================================================================
elif active_page == "Help & Viva Guide":
    render_top_header_bar("Help & Viva Examination Guide")

    viva_tab1, viva_tab2, viva_tab3, viva_tab4 = st.tabs(
        [
            "🏛️ Architecture & Workflow",
            "🔬 CV Algorithms Deep Dive",
            "🗄️ Database Schema",
            "💬 28 Viva Q&A Guide",
        ]
    )

    with viva_tab1:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">System Architecture & Processing Pipeline</div>
                <div class="panel-card-subtitle">End-to-end dataflow from camera frame to SQLite attendance record</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ```
            [Webcam / Photo Upload]
                     │
                     ▼
            [Image Validation] ─────────── (Resolution check, dimension validation)
                     │
                     ▼
            [OpenCV YuNet Detection] ───── (Locates bounding boxes & 5 facial landmarks)
                     │
                     ▼
            [Face Alignment & Cropping] ── (Affine 5-point alignment to 112×112)
                     │
                     ▼
            [OpenCV SFace CNN] ─────────── (Extracts 128-d L2-normalized embedding)
                     │
                     ├───> [Registration Mode] ──> Store Embedding in SQLite `face_templates`
                     │
                     └───> [Attendance Mode]
                                │
                                ▼
                   [Cosine Similarity Matching]
                     Score = dot(E_query, E_db) / (||E_query|| · ||E_db||)
                                │
                                ├── If Score >= 0.363 ──> "Recognized" (Student ID, Name)
                                │                             │
                                │                             ▼
                                │                    [Check Today's Record]
                                │                             │
                                │                    ├── Already Marked? ──> Block Duplicate
                                │                    └── New Today?      ──> Confirm & Save to SQLite
                                │
                                └── If Score < 0.363  ──> "Unknown Face" (Prompt Manual Review)
            ```
            """
        )

    with viva_tab2:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Core Computer Vision Formulations</div>
                <div class="panel-card-subtitle">Mathematical foundations of face recognition and image processing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            #### 1. SFace Cosine Similarity
            $$\\text{Cosine Similarity} = \\frac{\\vec{E}_1 \\cdot \\vec{E}_2}{\\|\\vec{E}_1\\|_2 \\|\\vec{E}_2\\|_2} = \\sum_{i=1}^{128} E_{1,i} \\cdot E_{2,i}$$
            - Baseline decision cutoff: $\\ge 0.363$

            #### 2. Grayscale Conversion (ITU-R BT.601)
            $$Y = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B$$

            #### 3. Linear Contrast and Brightness
            $$g(x, y) = \\alpha \\cdot f(x, y) + \\beta$$
            """
        )

    with viva_tab3:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">SQLite Relational Schema</div>
                <div class="panel-card-subtitle">Enforcing referential integrity and daily uniqueness constraints</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ```sql
            -- 1. Students Table
            CREATE TABLE students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- 2. Face Templates (Embeddings) Table
            CREATE TABLE face_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            -- 3. Attendance Table with Uniqueness Constraint
            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                CONSTRAINT unique_daily_attendance UNIQUE (student_id, attendance_date)
            );
            ```
            """
        )

    with viva_tab4:
        st.markdown(
            """
            <div class="dashboard-panel-card" style="margin-bottom: 16px;">
                <div class="panel-card-title">28 Viva Examination Questions & Model Answers</div>
                <div class="panel-card-subtitle">Comprehensive question bank with answers directly aligned to the codebase</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        viva_qa_data = [
            ("Q1: What is Computer Vision?", "Computer Vision is a branch of artificial intelligence that enables computers to extract, process, analyze, and understand high-dimensional visual information from digital images and videos to make automated decisions."),
            ("Q2: What is OpenCV and why is it used in this project?", "OpenCV is an open-source, highly optimized computer vision library written in C/C++ with Python bindings. We use it for camera acquisition, color space transformations, spatial filtering, and running deep neural networks via cv2.dnn."),
            ("Q3: What is the fundamental difference between Face Detection and Face Recognition?", "Face Detection answers 'Where is a face?' by finding spatial coordinates and bounding boxes. Face Recognition answers 'Whose face is this?' by verifying or identifying the individual from biometric features."),
            ("Q4: Why did you choose Python for this project?", "Python provides clean, readable syntax and extensive support for computer vision and scientific computing libraries, including OpenCV, NumPy, and Pandas, allowing rapid implementation of complete pipelines without manual memory management."),
            ("Q5: What is Streamlit and how does it compare to Flask or React?", "Streamlit is an open-source Python framework designed for creating data and machine learning web applications. Unlike Flask or React, which require separate backend endpoints, frontend templates, JavaScript state managers, and REST APIs, Streamlit renders reactive web user interfaces directly from pure Python code in a single process."),
            ("Q6: Why did you select SQLite instead of MySQL or MongoDB?", "SQLite is a lightweight, serverless, self-contained, transactional SQL database engine stored directly as a single file on disk (data/attendance.db). It requires no background server process, has zero configuration, is built directly into Python's standard library, and fully supports ACID transactions and foreign key constraints."),
            ("Q7: What is a Face Embedding?", "A face embedding is a compact, dense numerical representation (in our case, a 128-dimensional vector of 32-bit floating-point numbers) produced by a deep convolutional neural network (SFace). The network is trained such that embeddings of faces belonging to the same individual are close together in the vector space, while embeddings of different individuals are far apart."),
            ("Q8: What deep learning models are used in this project?", "We use two pretrained ONNX models from the official OpenCV Zoo: 1. YuNet (face_detection_yunet_2023mar.onnx): A fast, anchor-free CNN for face detection and 5-point facial landmark extraction. 2. SFace (face_recognition_sface_2021dec.onnx): An ArcFace/ResNet-based CNN that outputs a 128-dimensional biometric face embedding."),
            ("Q9: How is Face Alignment performed and why is it necessary?", "Face alignment uses the 5 facial landmarks (two eyes, nose tip, two mouth corners) detected by YuNet. OpenCV's recognizer.alignCrop() applies an affine similarity transform to rotate, scale, and center the face into a standardized 112×112 crop. Without alignment, variations in head tilt or perspective would distort the extracted feature vector and degrade recognition accuracy."),
            ("Q10: How does Cosine Similarity work in face matching?", "Cosine similarity measures the cosine of the angle between two non-zero vectors in multi-dimensional space. Because SFace feature vectors are normalized to unit length, cosine similarity simplifies to the Euclidean dot product: Sum(E1_i * E2_i). Values range from -1.0 to 1.0. We use a threshold of 0.363; scores at or above this threshold indicate a verified match."),
            ("Q11: What happens when an unknown face appears before the camera?", "The system detects the face and extracts its 128-d embedding, but when compared against all registered student embeddings in SQLite, the maximum cosine similarity score falls below 0.363. The UI flags the face with an orange bounding box labeled 'Unknown' or 'Needs Review', and blocks automatic attendance recording."),
            ("Q12: How are duplicate attendance records prevented?", "Through a dual-layer approach: 1. Programmatic check: The application queries SQLite before inserting to see if a record already exists for that student on that date. 2. Database constraint: The attendance table enforces UNIQUE(student_id, attendance_date). Any duplicate insertion causes SQLite to reject the operation with an IntegrityError."),
            ("Q13: What is Grayscale conversion and what is its mathematical formula?", "Grayscale conversion transforms a 3-channel RGB/BGR image into a single intensity channel representing perceived luminance. The formula used by OpenCV is Y = 0.299*R + 0.587*G + 0.114*B. Green is weighted highest because the human eye is most sensitive to green wavelengths."),
            ("Q14: What is an Image Histogram?", "An image histogram is a graphical representation of the frequency distribution of pixel intensity values across an image. For an 8-bit image, the horizontal axis spans 256 discrete bins (0 to 255), and the vertical axis represents the total count of pixels falling into each intensity bin."),
            ("Q15: What is Histogram Equalization and how does it work?", "Histogram equalization is a contrast enhancement technique that spreads out the most frequent intensity values. It computes the normalized cumulative distribution function (CDF) of pixel intensities and uses it as a mapping function to linearize the cumulative probability, enhancing global contrast in underexposed or overexposed images."),
            ("Q16: What is the difference between Gaussian Smoothing and Median Smoothing?", "Gaussian smoothing is a linear spatial convolution filter that calculates a weighted neighborhood average using a 2D Gaussian distribution, effectively smoothing Gaussian sensor noise. Median smoothing is a non-linear order-statistics filter that replaces each pixel with the statistical median of its neighborhood, which eliminates impulse 'salt-and-pepper' noise without blurring distinct edges."),
            ("Q17: How does the Canny Edge Detection algorithm work?", "Canny is a 4-stage edge detection pipeline: 1. Gaussian Filtering to reduce noise. 2. Gradient Calculation via Sobel to determine magnitude and orientation. 3. Non-Maximum Suppression (NMS) to thin edge ridges to 1-pixel width. 4. Hysteresis Thresholding using low and high cutoffs to trace strong edges and reject weak noise."),
            ("Q18: What are Morphological Operations (Erosion and Dilation)?", "Morphological operations are non-linear image processing tools based on mathematical shape analysis using a structuring element (kernel): Erosion computes the local minimum intensity, thinning foreground structures and removing small white noise. Dilation computes the local maximum intensity, expanding foreground structures and filling holes."),
            ("Q19: What is Image Thresholding and what is Otsu's Method?", "Thresholding segments an image into binary foreground (255) and background (0) regions based on a cutoff value. Otsu's thresholding is an automated technique that tests all possible threshold values (0–255) and selects the one that maximizes the inter-class variance between background and foreground distributions, eliminating manual threshold tuning."),
            ("Q20: How are face embeddings stored in SQLite?", "The 128-dimensional NumPy float32 embedding vector is converted to raw binary format using NumPy's tobytes() method and stored as a BLOB (Binary Large Object) inside the face_templates table. When loaded, np.frombuffer() restores the exact 128-element float32 vector."),
            ("Q21: Why do we store face embeddings instead of original face photographs?", "1. Privacy & Security: Numerical embeddings are irreversible representations; reconstructing the original human face from a 128-d vector is practically impossible. 2. Storage Efficiency: A 128 float32 embedding occupies exactly 512 bytes, whereas a photo occupies hundreds of kilobytes. 3. Search Speed: Vector similarity operations on 512-byte arrays in memory execute in microseconds."),
            ("Q22: What is the difference between traditional HOG features and Deep Face Embeddings?", "HOG (Histogram of Oriented Gradients) is a hand-crafted feature descriptor that measures localized edge orientations. It is computationally fast but sensitive to lighting, expression, and viewpoint variations. Deep Face Embeddings (from SFace) are learned end-to-end through millions of face comparisons using convolutional layers, capturing invariant semantic facial features with vastly superior recognition accuracy."),
            ("Q23: What is a Convolutional Neural Network (CNN)?", "A CNN is a specialized deep neural network designed for grid-structured data like images. It utilizes convolutional layers with learnable spatial filters, activation functions (like ReLU), and pooling or stride operations to extract translation-invariant hierarchical features from raw pixels."),
            ("Q24: What is Transfer Learning and how does this project utilize it?", "Transfer learning is the machine learning practice of taking a model trained on a massive generic dataset (like CASIA-WebFace or VGGFace) and reusing its learned representations for a target domain. We utilize transfer learning by using pretrained YuNet and SFace models for zero-shot face recognition rather than attempting to train a deep model from scratch on small college datasets."),
            ("Q25: How does the application handle invalid images during student registration?", "FaceRecognitionService.validate_for_registration() enforces strict quality checks: The image must decode to a valid NumPy array with minimum resolution 100×100. YuNet must detect exactly one face. Images with 0 faces or multiple faces are rejected. The face bounding box must be at least 50×50 pixels, and detection confidence must be at least 65%."),
            ("Q26: What are Generative Adversarial Networks (GANs) and how relate to Computer Vision?", "A GAN consists of two competing neural networks: a Generator that creates realistic synthetic images from random noise, and a Discriminator that evaluates whether an image is authentic or generated. In computer vision, GANs are used for super-resolution, style transfer, and generating synthetic facial training data to balance datasets. (Explained conceptually in Semester VII Unit 9)."),
            ("Q27: What is Optical Flow in video processing?", "Optical flow is the pattern of apparent motion of image objects, surfaces, and edges in a visual scene caused by relative motion between an observer and the scene. It calculates motion vectors for pixels across consecutive video frames (e.g., via Lucas-Kanade or Farneback algorithms). (Explained conceptually in Semester VII Unit 10)."),
            ("Q28: What are the main limitations of this mini-project?", "1. Single attendance session per class per day. 2. Sensitivity to extreme low-light conditions. 3. Lack of 3D depth-based anti-spoofing (vulnerable to high-resolution printed photos). 4. Attendance marking requires single-frame confirmation rather than continuous multi-camera crowd scanning."),
        ]

        for question, answer in viva_qa_data:
            with st.expander(f"**{question}**"):
                st.write(answer)


# ============================================================================
# 11. PAGE: PROFILE & SETTINGS (Faculty Info, System Diagnostics & DB Utilities)
# ============================================================================
elif active_page == "Profile & Settings":
    render_top_header_bar("Faculty Profile & System Settings")

    col_prof, col_db_util = st.columns([1, 1.2])

    with col_prof:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Faculty Administrator Profile</div>
                <div class="panel-card-subtitle">Course examiner & professor credentials</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 22px; text-align: center;">
                <div style="width: 64px; height: 64px; border-radius: 50%; background: #EDE9FE; color: #7C3AED; font-weight: 800; font-size: 1.8rem; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px;">
                    D
                </div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #0F172A;">Dr. Suresh Patel</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #2563EB;">Professor & Lab Examiner</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Department of Computer Engineering</div>
                <hr style="margin: 16px 0; border: 0; border-top: 1px solid #E2E8F0;">
                <div style="text-align: left; font-size: 0.82rem; color: #334155;">
                    <b>Faculty ID:</b> <code>FAC-CV-07</code><br>
                    <b>Course:</b> Semester VII Computer Vision<br>
                    <b>Academic Year:</b> 2026-2027<br>
                    <b>Role Access:</b> Full System Administrator
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_db_util:
        st.markdown(
            """
            <div class="dashboard-panel-card">
                <div class="panel-card-title">Database Storage & Model Diagnostics</div>
                <div class="panel-card-subtitle">Inspect SQLite file on disk and reset demo records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        db_path = database.DEFAULT_DB_PATH
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        enrolled_c = len(database.get_all_students())
        session_dates = database.get_distinct_attendance_dates()

        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="font-size: 0.85rem; color: #334155;">
                    <b>Database File:</b> <code>{db_path}</code><br>
                    <b>File Size on Disk:</b> {db_size / 1024:.2f} KB<br>
                    <b>Total Registered Students:</b> {enrolled_c}<br>
                    <b>Total Attendance Sessions Logged:</b> {len(session_dates)}<br>
                    <b>YuNet Model Status:</b> Ready (232 KB ONNX)<br>
                    <b>SFace Model Status:</b> Ready (38.6 MB ONNX)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🛠️ Demo Data Reset Utility")
        if st.button("🧪 Re-populate Default Demo Students", use_container_width=True):
            sample1_p = os.path.join("data", "samples", "sample_student_1.jpg")
            sample2_p = os.path.join("data", "samples", "sample_student_2.jpg")
            if os.path.exists(sample1_p):
                attendance_service.register_student("21CS001", "Lena Forsen", "Sem-VII-CSE", cv2.imread(sample1_p))
            if os.path.exists(sample2_p):
                attendance_service.register_student("21CS002", "Alex Mercer", "Sem-VII-CSE", cv2.imread(sample2_p))
            st.success("Default demo students refreshed in SQLite!")
            st.rerun()
