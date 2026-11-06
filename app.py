import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Union

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

# ============================================================
# 0. AUTO-CONFIGURE STREAMLIT LIGHT MEDICAL THEME
# ============================================================
os.makedirs(".streamlit", exist_ok=True)
config_toml_path = os.path.join(".streamlit", "config.toml")
if not os.path.exists(config_toml_path):
    try:
        with open(config_toml_path, "w", encoding="utf-8") as cf:
            cf.write(
                '[theme]\n'
                'base="light"\n'
                'primaryColor="#0284C7"\n'
                'backgroundColor="#F0F6FA"\n'
                'secondaryBackgroundColor="#FFFFFF"\n'
                'textColor="#0F172A"\n'
                'font="sans serif"\n'
            )
    except Exception:
        pass

load_dotenv()

st.set_page_config(
    page_title="NephroCare AI — Kidney Diagnostic Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. PRODUCTION MEDICAL UI CSS (CLEAR ROUNDED BOXES & LIGHT INPUTS)
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: light !important;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #EEF5FA !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1340px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 2px solid #D8E6F2 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Force ALL Streamlit Bordered Containers to look like clear rounded cards */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #FFFFFF !important;
        border: 2px solid #CFE0EE !important;
        border-radius: 22px !important;
        padding: 1.5rem 1.6rem !important;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05) !important;
    }

    /* CRITICAL FIX: Override BaseWeb dark input/textarea wrappers */
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 2px solid #BDD2E4 !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
    }

    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        caret-color: #0284C7 !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
        padding: 0.85rem 1rem !important;
        border-radius: 14px !important;
    }

    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 4px rgba(2, 132, 199, 0.15) !important;
    }

    /* Dropdowns / Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #BDD2E4 !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] section {
        background-color: #F8FBFF !important;
        border: 2px dashed #7DD3FC !important;
        border-radius: 18px !important;
        padding: 1.2rem !important;
        color: #0F172A !important;
    }

    /* Radio Button Group */
    div[role="radiogroup"] {
        background: #F0F7FC !important;
        padding: 0.5rem 1rem !important;
        border-radius: 14px !important;
        border: 1.5px solid #CFE0EE !important;
    }
    div[role="radiogroup"] label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* Input Labels */
    .stTextArea label p,
    .stTextInput label p,
    .stNumberInput label p,
    .stSelectbox label p,
    .stFileUploader label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* Secondary Preset Buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #F0F7FC !important;
        color: #0369A1 !important;
        border: 2px solid #BAE6FD !important;
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.18s ease !important;
    }
    div.stButton > button[kind="secondary"] * {
        color: #0369A1 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #E0F2FE !important;
        border-color: #0284C7 !important;
        transform: translateY(-1px) !important;
    }

    /* Primary Action Button */
    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0.85rem 1.5rem !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.28) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"] *,
    div[data-testid="stFormSubmitButton"] > button * {
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 26px rgba(2, 132, 199, 0.38) !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: #0284C7 !important;
        background-color: #F0F9FF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 22px rgba(2, 132, 199, 0.16) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 2px solid #D8E6F2 !important;
        border-radius: 14px !important;
        padding: 0.65rem 1.35rem !important;
        margin-right: 0.65rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border-color: #0284C7 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    /* ========================================================
       SELF-CONTAINED ROUNDED MEDICAL CARDS (100% RELIABLE)
       ======================================================== */
    .top-banner {
        background: linear-gradient(135deg, #075985 0%, #0D9488 100%);
        border-radius: 24px;
        padding: 1.7rem 2.2rem;
        color: #FFFFFF;
        box-shadow: 0 12px 30px rgba(7, 89, 133, 0.18);
        margin-bottom: 1.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .top-banner h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }
    .top-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
        color: #E0F2FE !important;
        font-weight: 500;
    }
    .status-chip {
        background: rgba(255, 255, 255, 0.18);
        border: 1.5px solid rgba(255, 255, 255, 0.35);
        color: #FFFFFF !important;
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Rounded Box Header Tag */
    .pill-tag {
        display: inline-block;
        background: #E0F2FE;
        color: #0369A1;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 0.25rem 0;
    }
    .card-sub {
        font-size: 0.9rem;
        color: #64748B;
        margin: 0 0 1rem 0;
    }

    .readable-paragraph {
        font-size: 0.98rem;
        line-height: 1.72;
        color: #1E293B;
        margin: 0.4rem 0;
    }
    .readable-list {
        margin: 0.4rem 0 0 1.2rem;
        padding: 0;
        font-size: 0.95rem;
        line-height: 1.68;
        color: #1E293B;
    }
    .readable-list li {
        margin-bottom: 0.35rem;
    }

    /* ========================================================
       ✨ PREMIUM SUMMARY STAGE (glow background + soft shadows)
       ======================================================== */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(22px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatGlow {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50%      { transform: translate(24px, -18px) scale(1.1); }
    }
    @keyframes ringPop {
        from { opacity: 0; transform: scale(0.8) rotate(-30deg); }
        to   { opacity: 1; transform: scale(1) rotate(0deg); }
    }
    @keyframes softPulse {
        0%, 100% { box-shadow: 0 0 0 0 var(--pulse, rgba(2,132,199,0.35)); }
        50%      { box-shadow: 0 0 0 10px rgba(255,255,255,0); }
    }

    .summary-stage {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        margin-top: 0.6rem;
        padding: 1.7rem;
        border-radius: 34px;
        background: linear-gradient(145deg, #F8FCFF 0%, #EAF4FB 52%, #E4F6F2 100%);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        box-shadow:
            0 40px 70px -28px rgba(7, 89, 133, 0.34),
            0 18px 32px -16px rgba(15, 23, 42, 0.14),
            inset 0 1px 0 rgba(255, 255, 255, 1);
        animation: fadeUp 0.65s cubic-bezier(.2,.8,.2,1) both;
    }
    .summary-stage::before,
    .summary-stage::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        filter: blur(60px);
        z-index: -1;
        pointer-events: none;
    }
    .summary-stage::before {
        width: 380px; height: 380px;
        top: -140px; left: -110px;
        background: radial-gradient(circle, rgba(56,189,248,0.42) 0%, rgba(56,189,248,0) 70%);
        animation: floatGlow 11s ease-in-out infinite;
    }
    .summary-stage::after {
        width: 420px; height: 420px;
        bottom: -170px; right: -120px;
        background: radial-gradient(circle, rgba(45,212,191,0.38) 0%, rgba(45,212,191,0) 70%);
        animation: floatGlow 14s ease-in-out infinite reverse;
    }

    /* Hero strip inside the stage */
    .sum-hero {
        display: flex;
        align-items: center;
        gap: 1.6rem;
        flex-wrap: wrap;
        padding: 1.4rem 1.7rem;
        border-radius: 26px;
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        box-shadow:
            0 22px 44px -18px rgba(15, 23, 42, 0.18),
            0 6px 14px -6px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.4rem;
        position: relative;
    }
    .sum-hero::before {
        content: "";
        position: absolute;
        left: 0; top: 18px; bottom: 18px;
        width: 7px;
        border-radius: 0 8px 8px 0;
    }
    .sum-hero-ckd::before { background: linear-gradient(180deg, #FB7185, #E11D48); }
    .sum-hero-ok::before  { background: linear-gradient(180deg, #34D399, #059669); }

    .conf-ring {
        width: 128px; height: 128px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        animation: ringPop 0.9s cubic-bezier(.2,.9,.3,1.2) both;
        box-shadow: 0 16px 30px -10px rgba(15, 23, 42, 0.28);
    }
    .conf-ring-inner {
        width: 96px; height: 96px;
        border-radius: 50%;
        background: #FFFFFF;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        box-shadow: inset 0 3px 8px rgba(15, 23, 42, 0.08);
    }
    .conf-ring-inner strong {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .conf-ring-inner span {
        font-size: 0.62rem;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .sum-hero-text { flex: 1; min-width: 260px; }
    .risk-chip {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 0.32rem 0.9rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-chip-ckd { background: #FFE4E6; color: #BE123C; }
    .risk-chip-ok  { background: #D1FAE5; color: #047857; }
    .sum-hero-text h2 {
        margin: 0.6rem 0 0.7rem 0;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.015em;
    }
    .sum-hero-ckd h2 { color: #881337 !important; }
    .sum-hero-ok h2  { color: #064E3B !important; }

    .meta-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .meta-chip {
        background: #F1F5F9;
        border: 1.5px solid #E2E8F0;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
    }
    .meta-chip-warn {
        background: #FFF1F2;
        border-color: #FDA4AF;
        color: #9F1239;
    }
    .meta-chip-good {
        background: #ECFDF5;
        border-color: #6EE7B7;
        color: #065F46;
    }

    /* Doctor / Patient grid */
    .sum-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
        gap: 1.4rem;
    }
    .sum-card {
        position: relative;
        display: flex;
        flex-direction: column;
        padding: 1.7rem 1.8rem 1.6rem;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.92);
        border: 1.5px solid rgba(255, 255, 255, 1);
        overflow: hidden;
        transition: transform 0.35s cubic-bezier(.2,.8,.2,1), box-shadow 0.35s ease;
        animation: fadeUp 0.7s cubic-bezier(.2,.8,.2,1) both;
    }
    .sum-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 6px;
    }
    .sum-card::after {
        content: "";
        position: absolute;
        width: 220px; height: 220px;
        top: -110px; right: -90px;
        border-radius: 50%;
        opacity: 0.55;
        pointer-events: none;
    }
    .sum-card-doc {
        animation-delay: 0.12s;
        box-shadow:
            0 30px 50px -24px rgba(2, 132, 199, 0.38),
            0 12px 22px -12px rgba(15, 23, 42, 0.12);
    }
    .sum-card-doc::before { background: linear-gradient(90deg, #0284C7, #38BDF8); }
    .sum-card-doc::after  { background: radial-gradient(circle, rgba(186,230,253,0.9), rgba(186,230,253,0) 70%); }
    .sum-card-pat {
        animation-delay: 0.24s;
        box-shadow:
            0 30px 50px -24px rgba(13, 148, 136, 0.38),
            0 12px 22px -12px rgba(15, 23, 42, 0.12);
    }
    .sum-card-pat::before { background: linear-gradient(90deg, #0D9488, #2DD4BF); }
    .sum-card-pat::after  { background: radial-gradient(circle, rgba(153,246,228,0.9), rgba(153,246,228,0) 70%); }

    .sum-card-doc:hover {
        transform: translateY(-8px);
        box-shadow:
            0 44px 70px -26px rgba(2, 132, 199, 0.5),
            0 18px 30px -14px rgba(15, 23, 42, 0.16);
    }
    .sum-card-pat:hover {
        transform: translateY(-8px);
        box-shadow:
            0 44px 70px -26px rgba(13, 148, 136, 0.5),
            0 18px 30px -14px rgba(15, 23, 42, 0.16);
    }

    .sum-card-head {
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        z-index: 1;
        margin-bottom: 0.4rem;
    }
    .sum-icon {
        width: 58px; height: 58px;
        flex-shrink: 0;
        border-radius: 18px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.7rem;
        box-shadow: 0 12px 22px -8px rgba(15, 23, 42, 0.3);
        transition: transform 0.35s ease;
    }
    .sum-card:hover .sum-icon { transform: rotate(-6deg) scale(1.08); }
    .sum-icon-doc { background: linear-gradient(135deg, #38BDF8, #0284C7); }
    .sum-icon-pat { background: linear-gradient(135deg, #5EEAD4, #0D9488); }

    .sum-card .card-sub { position: relative; z-index: 1; margin-bottom: 0.6rem; }
    .sum-body { position: relative; z-index: 1; }

    .sum-section {
        position: relative;
        z-index: 1;
        margin-top: 1.1rem;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
        border: 1.5px solid #E2E8F0;
        border-left: 5px solid #0284C7;
    }
    .sum-section-warn {
        background: linear-gradient(135deg, #FFF7F8, #FFF1F2);
        border-color: #FECDD3;
        border-left-color: #E11D48;
    }
    .sum-section-teal {
        background: linear-gradient(135deg, #F7FEFC, #F0FDFA);
        border-color: #CCFBF1;
        border-left-color: #0D9488;
    }
    .sum-section-title {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    /* Custom bullets inside summary cards */
    .sum-card .readable-list,
    .history-card .readable-list {
        list-style: none;
        margin: 0.4rem 0 0 0;
        padding: 0;
    }
    .sum-card .readable-list li,
    .history-card .readable-list li {
        position: relative;
        padding-left: 1.85rem;
        margin-bottom: 0.5rem;
    }
    .sum-card .readable-list li::before,
    .history-card .readable-list li::before {
        content: "✓";
        position: absolute;
        left: 0; top: 0.18rem;
        width: 1.2rem; height: 1.2rem;
        border-radius: 50%;
        background: #DCFCE7;
        color: #166534;
        font-size: 0.7rem;
        font-weight: 800;
        display: flex; align-items: center; justify-content: center;
    }
    .sum-section-warn .readable-list li::before {
        content: "!";
        background: #FFE4E6;
        color: #BE123C;
    }
    .sum-section-teal .readable-list li::before {
        background: #CCFBF1;
        color: #0F766E;
    }
    .sum-card .readable-paragraph { margin: 0.5rem 0; }

    /* Step 2 Snapshot Card */
    .snapshot-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAFDFF 100%);
        border: 2px solid #CFE0EE;
        border-radius: 24px;
        padding: 1.5rem 1.6rem;
        box-shadow:
            0 26px 44px -22px rgba(2, 132, 199, 0.28),
            0 10px 20px -10px rgba(15, 23, 42, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeUp 0.6s ease both;
    }
    .snapshot-card:hover {
        transform: translateY(-4px);
        box-shadow:
            0 36px 56px -24px rgba(2, 132, 199, 0.38),
            0 14px 26px -12px rgba(15, 23, 42, 0.1);
    }
    .patient-banner {
        background: linear-gradient(135deg, #F0F9FF, #E6F7F4);
        border: 2px solid #BAE6FD;
        border-radius: 16px;
        padding: 0.95rem 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .lab-mini-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.7rem;
    }
    .lab-pill-normal,
    .lab-pill-abnormal {
        border-radius: 14px;
        padding: 0.75rem 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .lab-pill-normal {
        background: #F8FAFC;
        border: 1.5px solid #E2E8F0;
    }
    .lab-pill-abnormal {
        background: #FFF1F2;
        border: 1.5px solid #FDA4AF;
    }
    .lab-pill-normal:hover,
    .lab-pill-abnormal:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 18px -8px rgba(15, 23, 42, 0.18);
    }

    /* 20-Biomarker Grid */
    .bio-panel-box {
        background: #FFFFFF;
        border: 2px solid #CFE0EE;
        border-radius: 26px;
        padding: 1.6rem 1.75rem;
        margin-top: 1.4rem;
        box-shadow:
            0 26px 44px -24px rgba(15, 23, 42, 0.18),
            0 10px 20px -10px rgba(15, 23, 42, 0.06);
        animation: fadeUp 0.8s ease both;
    }
    .bio-grid-20 {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(195px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .bio-tile-ok,
    .bio-tile-flag {
        border-radius: 16px;
        padding: 0.85rem 1rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .bio-tile-ok {
        background: #F8FAFC;
        border: 1.5px solid #E2E8F0;
    }
    .bio-tile-flag {
        background: #FFF1F2;
        border: 2px solid #FDA4AF;
    }
    .bio-tile-ok:hover,
    .bio-tile-flag:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 24px -10px rgba(15, 23, 42, 0.2);
    }
    .badge-ok {
        background: #DCFCE7;
        color: #166534;
        font-size: 0.7rem;
        font-weight: 800;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
    }
    .badge-flag {
        background: #FFE4E6;
        color: #9F1239;
        font-size: 0.7rem;
        font-weight: 800;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
    }

    /* Saved reports (history) cards */
    .history-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1.5px solid #D8E6F2;
        border-radius: 24px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow:
            0 26px 44px -24px rgba(2, 132, 199, 0.26),
            0 10px 20px -10px rgba(15, 23, 42, 0.07);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeUp 0.6s ease both;
    }
    .history-card:hover {
        transform: translateY(-5px);
        box-shadow:
            0 38px 58px -26px rgba(2, 132, 199, 0.4),
            0 14px 26px -12px rgba(15, 23, 42, 0.1);
    }

    /* ========================================================
       🔀 PIPELINE FLOW (data → LLM → ML trees → LLM → report)
       ======================================================== */
    @keyframes packetMove {
        0%   { left: 0; opacity: 0; }
        15%  { opacity: 1; }
        85%  { opacity: 1; }
        100% { left: calc(100% - 12px); opacity: 0; }
    }
    @keyframes nodeScan {
        0%   { top: -8px; opacity: 0; }
        15%  { opacity: 1; }
        85%  { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
    @keyframes treeBlink {
        0%, 100% { transform: translateY(0) scale(1);   opacity: 0.45; }
        50%      { transform: translateY(-4px) scale(1.25); opacity: 1; }
    }

    .ai-think.flow-panel { display: block; padding: 1.4rem 1.5rem 1.2rem; }
    .flow-panel::after { display: none !important; }
    .flow-head { display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.9rem; }
    .flow-head .ai-body { flex: 1; min-width: 0; }
    .ai-core.mini { width: 62px; height: 62px; font-size: 1.7rem; flex-shrink: 0; }
    .ai-pct.big { font-size: 1.35rem; min-width: 64px; }

    .fl-row {
        display: flex;
        align-items: center;
        gap: 0.15rem;
        margin-top: 1.1rem;
        padding: 1rem 0.7rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.78);
        border: 1.5px solid rgba(255, 255, 255, 1);
        box-shadow: 0 18px 34px -22px rgba(15, 23, 42, 0.28);
        overflow-x: auto;
    }
    .fl-node {
        position: relative;
        overflow: hidden;
        flex: 0 0 92px;
        text-align: center;
        padding: 0.7rem 0.3rem 0.6rem;
        border-radius: 18px;
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    .fl-ico {
        position: relative;
        width: 46px; height: 46px;
        margin: 0 auto 0.35rem;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        background: #F1F5F9;
        filter: grayscale(0.6);
        opacity: 0.7;
    }
    .fl-name { font-size: 0.74rem; font-weight: 800; color: #0F172A; }
    .fl-sub  { font-size: 0.64rem; font-weight: 600; color: #64748B; min-height: 1.1rem; }

    .fl-node.active {
        border-color: #38BDF8;
        transform: translateY(-4px);
        box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.22), 0 18px 30px -12px rgba(2, 132, 199, 0.6);
    }
    .fl-node.active .fl-ico {
        background: linear-gradient(135deg, #0284C7, #0D9488);
        filter: none; opacity: 1;
        animation: orbPulse 1.1s ease-in-out infinite;
    }
    .fl-node.active .fl-sub { color: #0369A1; }
    .fl-node.scanning::after {
        content: "";
        position: absolute;
        left: 0; right: 0;
        height: 8px;
        background: linear-gradient(180deg, rgba(45,212,191,0), rgba(45,212,191,0.9));
        animation: nodeScan 1.1s ease-in-out infinite;
    }
    .fl-node.done { border-color: #6EE7B7; background: #F0FDF9; }
    .fl-node.done .fl-ico { background: #D1FAE5; filter: none; opacity: 1; }
    .fl-node.done .fl-ico::after {
        content: "✓";
        position: absolute;
        right: -7px; bottom: -7px;
        width: 18px; height: 18px;
        border-radius: 50%;
        background: #059669;
        color: #FFFFFF;
        font-size: 0.65rem; font-weight: 800;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 8px rgba(5, 150, 105, 0.4);
    }
    .fl-trees { display: inline-flex; gap: 1px; font-size: 0.7rem; }
    .fl-trees b { font-weight: 400; animation: treeBlink 0.8s ease-in-out infinite; }
    .fl-trees b:nth-child(2) { animation-delay: 0.12s; }
    .fl-trees b:nth-child(3) { animation-delay: 0.24s; }
    .fl-trees b:nth-child(4) { animation-delay: 0.36s; }
    .fl-trees b:nth-child(5) { animation-delay: 0.48s; }

    .fl-conn { position: relative; flex: 1 1 26px; min-width: 26px; height: 12px; }
    .fl-conn::before {
        content: "";
        position: absolute;
        left: 0; right: 0; top: 4px;
        height: 4px;
        border-radius: 4px;
        background: #DCEBF6;
    }
    .fl-conn.done::before { background: linear-gradient(90deg, #34D399, #10B981); }
    .fl-conn.sending::before {
        background: linear-gradient(90deg, #BAE6FD, #38BDF8, #BAE6FD);
        background-size: 200% 100%;
        animation: borderFlow 0.7s linear infinite;
    }
    .fl-conn i {
        position: absolute;
        top: 0; left: 0;
        width: 12px; height: 12px;
        border-radius: 50%;
        background: radial-gradient(circle, #5EEAD4, #0284C7);
        box-shadow: 0 0 10px rgba(2, 132, 199, 0.85);
        opacity: 0;
    }
    .fl-conn.sending i { animation: packetMove 0.9s linear infinite; }
    .fl-conn.sending i:nth-child(2) { animation-delay: 0.3s; }
    .fl-conn.sending i:nth-child(3) { animation-delay: 0.6s; }

    @media (prefers-reduced-motion: reduce) {
        .fl-conn i, .fl-node.scanning::after, .fl-trees b, .fl-node.active .fl-ico { animation: none !important; }
    }

    /* ========================================================
       🔍 POPUP (st.dialog) + SCAN BEAM + PATIENT SCREEN
       ======================================================== */
    @keyframes scanDown {
        0%   { top: -70px; opacity: 0; }
        12%  { opacity: 1; }
        88%  { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
    @keyframes ctaGlow {
        0%, 100% { box-shadow: 0 14px 30px -12px rgba(5, 150, 105, 0.45); }
        50%      { box-shadow: 0 18px 38px -10px rgba(5, 150, 105, 0.75); }
    }
    .ai-think::after {
        content: "";
        position: absolute;
        left: 0; right: 0;
        height: 70px;
        pointer-events: none;
        background: linear-gradient(180deg, rgba(56,189,248,0) 0%, rgba(56,189,248,0.3) 60%, rgba(45,212,191,0.55) 100%);
        border-bottom: 2px solid rgba(2, 132, 199, 0.55);
        animation: scanDown 2.6s ease-in-out infinite;
    }
    .ai-think-done::after { display: none; }
    .ai-think-done { border-color: #6EE7B7; }

    div[data-testid="stDialog"] div[role="dialog"] {
        border-radius: 30px !important;
        background: #F4F9FD !important;
        border: 2px solid #BAE6FD !important;
        box-shadow: 0 50px 110px -24px rgba(7, 89, 133, 0.55) !important;
    }
    div[data-testid="stDialog"] h2 {
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.01em;
    }
    .cta-note {
        margin: 1rem 0 0.7rem 0;
        padding: 1rem 1.3rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #ECFDF5, #F0FDFA);
        border: 2px solid #6EE7B7;
        color: #065F46;
        font-size: 1.02rem;
        font-weight: 700;
        text-align: center;
        animation: fadeUp 0.5s ease both;
    }

    .pt-hero {
        padding: 1.5rem 1.9rem;
        margin-bottom: 0.4rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #FFFFFF 0%, #EAF4FB 100%);
        border: 2px solid #BAE6FD;
        box-shadow: 0 26px 46px -24px rgba(2, 132, 199, 0.4), 0 10px 20px -10px rgba(15, 23, 42, 0.08);
        animation: fadeUp 0.5s ease both;
    }
    .pt-hero h2 { margin: 0.3rem 0 0.2rem 0; font-size: 1.55rem; font-weight: 800; color: #0F172A !important; }
    .pt-hero p  { margin: 0; color: #64748B; font-size: 0.95rem; font-weight: 500; }

    .patient-para {
        font-size: 1.04rem !important;
        line-height: 1.85 !important;
        color: #0F172A !important;
    }

    /* ========================================================
       🧠 LIVE "AI IS THINKING" PANEL
       ======================================================== */
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes orbPulse {
        0%, 100% { transform: scale(1); }
        50%      { transform: scale(1.07); }
    }
    @keyframes ringExpand {
        0%   { transform: scale(0.75); opacity: 0.75; }
        100% { transform: scale(1.75); opacity: 0; }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes dotBounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
        40%           { transform: translateY(-6px); opacity: 1; }
    }
    @keyframes thoughtCycle {
        0%        { opacity: 0; transform: translateY(10px); }
        5%, 22%   { opacity: 1; transform: translateY(0); }
        28%, 100% { opacity: 0; transform: translateY(-10px); }
    }
    @keyframes chipPop {
        from { opacity: 0; transform: scale(0.7) translateY(10px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }
    @keyframes borderFlow {
        0%   { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    .ai-think {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        display: flex;
        gap: 1.8rem;
        flex-wrap: wrap;
        align-items: stretch;
        margin-top: 1.2rem;
        padding: 1.8rem 2rem;
        border-radius: 30px;
        background: linear-gradient(135deg, #F8FCFF 0%, #EAF4FB 55%, #E4F6F2 100%);
        border: 2px solid #BAE6FD;
        box-shadow:
            0 36px 64px -28px rgba(2, 132, 199, 0.45),
            0 14px 26px -14px rgba(15, 23, 42, 0.14),
            inset 0 1px 0 rgba(255, 255, 255, 1);
        animation: fadeUp 0.45s cubic-bezier(.2,.8,.2,1) both;
    }
    .ai-think::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        background: linear-gradient(110deg, rgba(56,189,248,0.22), rgba(45,212,191,0.22), rgba(56,189,248,0.22));
        background-size: 200% 100%;
        animation: shimmer 6s linear infinite;
    }

    .ai-main {
        display: flex;
        gap: 1.6rem;
        align-items: center;
        flex: 1.4;
        min-width: 340px;
    }
    .ai-orb-wrap {
        position: relative;
        width: 150px; height: 150px;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
    }
    .ai-core {
        position: relative;
        z-index: 3;
        width: 86px; height: 86px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0284C7, #0D9488);
        display: flex; align-items: center; justify-content: center;
        font-size: 2.3rem;
        box-shadow:
            0 20px 36px -10px rgba(2, 132, 199, 0.65),
            inset 0 2px 6px rgba(255, 255, 255, 0.45);
        animation: orbPulse 2.2s ease-in-out infinite;
    }
    .ai-core-done {
        background: linear-gradient(135deg, #10B981, #059669);
        animation: none;
    }
    .ai-pulse {
        position: absolute;
        inset: 26px;
        border-radius: 50%;
        border: 2px solid rgba(2, 132, 199, 0.5);
        animation: ringExpand 2.4s ease-out infinite;
    }
    .ai-pulse.p2 { animation-delay: 1.2s; }
    .ai-orbit {
        position: absolute;
        inset: 0;
        z-index: 2;
        animation: spin 3.2s linear infinite;
    }
    .ai-orbit::before {
        content: "";
        position: absolute;
        inset: 9px;
        border-radius: 50%;
        border: 1.5px dashed rgba(2, 132, 199, 0.4);
    }
    .ai-orbit i {
        position: absolute;
        width: 13px; height: 13px;
        border-radius: 50%;
        background: linear-gradient(135deg, #38BDF8, #2DD4BF);
        box-shadow: 0 0 14px rgba(45, 212, 191, 0.95);
    }
    .ai-orbit i:nth-child(1) { top: 2px;  left: calc(50% - 6px); }
    .ai-orbit i:nth-child(2) { bottom: 22px; right: 16px; }
    .ai-orbit i:nth-child(3) { bottom: 22px; left: 16px; }

    .ai-body { flex: 1; min-width: 220px; }
    .ai-title {
        margin: 0.1rem 0 0.2rem 0;
        font-size: 1.3rem;
        font-weight: 800;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .ai-dots { display: inline-flex; gap: 4px; }
    .ai-dots b {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #0284C7;
        display: inline-block;
        animation: dotBounce 1.2s ease-in-out infinite;
    }
    .ai-dots b:nth-child(2) { animation-delay: 0.15s; background: #0891B2; }
    .ai-dots b:nth-child(3) { animation-delay: 0.3s;  background: #0D9488; }
    .ai-detail {
        margin: 0 0 0.9rem 0;
        font-size: 0.92rem;
        color: #475569;
        font-weight: 500;
    }
    .ai-bar-row { display: flex; align-items: center; gap: 0.8rem; }
    .ai-bar {
        flex: 1;
        height: 12px;
        border-radius: 999px;
        background: #DCEBF6;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(15, 23, 42, 0.1);
    }
    .ai-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #0284C7, #2DD4BF, #0284C7);
        background-size: 200% 100%;
        animation: borderFlow 1.6s linear infinite;
        box-shadow: 0 0 14px rgba(2, 132, 199, 0.55);
    }
    .ai-pct {
        font-size: 0.85rem;
        font-weight: 800;
        color: #0369A1;
        min-width: 42px;
        text-align: right;
    }
    .ai-thoughts {
        position: relative;
        height: 1.7rem;
        margin-top: 0.9rem;
        font-size: 0.9rem;
        font-style: italic;
        color: #0F766E;
        font-weight: 600;
    }
    .ai-thoughts span {
        position: absolute;
        left: 0; top: 0;
        opacity: 0;
        animation: thoughtCycle 12s linear infinite both;
    }
    .ai-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.5rem; }
    .ai-chip {
        background: #FFFFFF;
        border: 1.5px solid #BAE6FD;
        color: #0369A1;
        font-size: 0.8rem;
        font-weight: 800;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        box-shadow: 0 8px 14px -8px rgba(2, 132, 199, 0.4);
        animation: chipPop 0.5s cubic-bezier(.2,.9,.3,1.3) both;
    }
    .ai-chip-flag { background: #FFF1F2; border-color: #FDA4AF; color: #BE123C; }

    .ai-steps {
        list-style: none;
        margin: 0;
        padding: 1.1rem 1.3rem;
        flex: 1;
        min-width: 280px;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1.5px solid rgba(255, 255, 255, 1);
        box-shadow: 0 18px 34px -20px rgba(15, 23, 42, 0.25);
    }
    .ai-steps li {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.5rem 0.3rem;
        font-size: 0.92rem;
        font-weight: 700;
        color: #94A3B8;
        transition: color 0.3s ease;
    }
    .ai-ico {
        width: 28px; height: 28px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem;
        font-weight: 800;
        background: #E2E8F0;
        color: #94A3B8;
    }
    .ai-steps li.done { color: #047857; }
    .ai-steps li.done .ai-ico { background: #D1FAE5; color: #047857; }
    .ai-steps li.active { color: #0F172A; }
    .ai-steps li.active .ai-ico {
        background: transparent;
        border: 3px solid #BAE6FD;
        border-top-color: #0284C7;
        animation: spin 0.8s linear infinite;
        color: transparent;
    }

    /* ========================================================
       📚 PATIENT REGISTRY TABLE (history + predictions)
       ======================================================== */
    .reg-wrap {
        background: #FFFFFF;
        border: 2px solid #CFE0EE;
        border-radius: 26px;
        overflow: hidden;
        margin-top: 0.4rem;
        box-shadow:
            0 30px 50px -26px rgba(2, 132, 199, 0.32),
            0 12px 22px -12px rgba(15, 23, 42, 0.08);
        animation: fadeUp 0.6s ease both;
    }
    .reg-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        padding: 1.2rem 1.5rem 0.9rem;
        background: linear-gradient(135deg, #F8FCFF, #EEF7FB);
        border-bottom: 1.5px solid #E2E8F0;
    }
    .reg-head .card-title { margin: 0; }
    .reg-count {
        background: #E0F2FE;
        color: #0369A1;
        font-size: 0.82rem;
        font-weight: 800;
        padding: 0.4rem 1rem;
        border-radius: 999px;
    }
    .reg-scroll { overflow: auto; max-height: 560px; }
    table.reg-table {
        border-collapse: separate;
        border-spacing: 0;
        width: max-content;
        min-width: 100%;
        font-size: 0.86rem;
        color: #1E293B;
    }
    .reg-table th, .reg-table td {
        padding: 0.7rem 0.95rem;
        white-space: nowrap;
        text-align: left;
        border-bottom: 1px solid #E8EFF6;
    }
    .reg-table thead th {
        position: sticky;
        z-index: 3;
        background: #F0F7FC;
        color: #0F172A;
        font-weight: 800;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 2px solid #CFE0EE;
    }
    .reg-table thead small {
        display: block;
        font-size: 0.66rem;
        font-weight: 600;
        color: #64748B;
        text-transform: none;
        letter-spacing: 0;
    }
    .reg-table thead tr.grp th {
        top: 0;
        height: 40px;
        text-align: center;
        font-size: 0.78rem;
        border-bottom: 1.5px solid #CFE0EE;
    }
    .reg-table thead tr.cols th { top: 41px; }
    th.g-report { background: #E0F2FE !important; color: #075985; }
    th.g-pred   { background: #FFE4E6 !important; color: #9F1239; }
    th.g-data   { background: #CCFBF1 !important; color: #0F766E; }

    .reg-table tbody td { background: #FFFFFF; transition: background 0.15s ease; }
    .reg-table tbody tr:nth-child(even) td { background: #F8FBFE; }
    .reg-table tbody tr:hover td { background: #E6F4FD; }
    .reg-table td.cell-flag {
        background: #FFF1F2 !important;
        color: #BE123C;
        font-weight: 800;
    }
    .reg-table .sticky-1, .reg-table .sticky-2 { position: sticky; z-index: 2; }
    .reg-table thead .sticky-1, .reg-table thead .sticky-2 { z-index: 5; }
    .reg-table .sticky-1 { left: 0; min-width: 70px; max-width: 70px; }
    .reg-table .sticky-2 {
        left: 70px;
        min-width: 170px;
        box-shadow: 8px 0 10px -8px rgba(15, 23, 42, 0.18);
    }
    .reg-table .id-cell   { font-weight: 800; color: #0369A1; }
    .reg-table .name-cell { font-weight: 800; color: #0F172A; }
    .reg-table .date-cell { color: #64748B; font-weight: 600; }

    .tbl-badge {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
    }
    .tbl-badge-ckd { background: #FFE4E6; color: #BE123C; border: 1.5px solid #FDA4AF; }
    .tbl-badge-ok  { background: #D1FAE5; color: #047857; border: 1.5px solid #6EE7B7; }
    .conf-mini { display: flex; align-items: center; gap: 0.55rem; min-width: 130px; }
    .conf-mini-track {
        flex: 1;
        height: 8px;
        border-radius: 999px;
        background: #E2E8F0;
        overflow: hidden;
    }
    .conf-mini-fill { height: 100%; border-radius: 999px; }
    .conf-mini-fill.ckd { background: linear-gradient(90deg, #FB7185, #E11D48); }
    .conf-mini-fill.ok  { background: linear-gradient(90deg, #34D399, #059669); }
    .conf-mini span { font-weight: 800; font-size: 0.82rem; }

    @media (prefers-reduced-motion: reduce) {
        .ai-think, .ai-think::before, .ai-core, .ai-pulse, .ai-orbit,
        .ai-dots b, .ai-bar-fill, .ai-chip, .reg-wrap {
            animation: none !important;
        }
        .ai-thoughts span:first-child { opacity: 1; }
    }
    @media (max-width: 700px) {
        .ai-think { padding: 1.2rem; }
        .ai-main { min-width: 100%; }
    }

    @media (prefers-reduced-motion: reduce) {
        .summary-stage, .summary-stage::before, .summary-stage::after,
        .sum-card, .conf-ring, .snapshot-card, .bio-panel-box, .history-card {
            animation: none !important;
        }
    }
    @media (max-width: 700px) {
        .sum-grid { grid-template-columns: 1fr; }
        .summary-stage { padding: 1rem; border-radius: 26px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. DATA SCHEMA & CONSTANTS (EXACT FROM PROJECT_1.PY)
# ============================================================
class Extractor(BaseModel):
    name: Optional[str] = None
    age: Optional[Union[int, float]] = None
    bp: Optional[Union[int, float]] = None
    al: Optional[Union[int, float]] = None
    su: Optional[Union[int, float]] = None
    rbc: Optional[str] = None
    pc: Optional[str] = None
    pcc: Optional[str] = None
    ba: Optional[str] = None
    bgr: Optional[Union[int, float]] = None
    bu: Optional[Union[int, float]] = None
    sc: Optional[Union[int, float]] = None
    hemo: Optional[Union[int, float]] = None
    pcv: Optional[Union[int, float]] = None
    rc: Optional[Union[int, float]] = None
    htn: Optional[str] = None
    dm: Optional[str] = None
    cad: Optional[str] = None
    appet: Optional[str] = None
    pe: Optional[str] = None
    ane: Optional[str] = None


DEFAULT_MODEL_PATH = r"D:\RAG\InnovateX\kidney_model.pkl"
FALLBACK_MODEL_PATH = "kidney_model.pkl"
JSON_PATH = "patient_reports.json"

FEATURES = [
    "age", "bp", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu",
    "sc", "hemo", "pcv", "rc", "htn", "dm", "cad", "appet", "pe", "ane",
]

FEATURE_META = {
    "age":   ("Age", "years", "Adult"),
    "bp":    ("Blood Pressure", "mm/Hg", "70–85"),
    "al":    ("Urine Albumin", "scale 0–5", "0"),
    "su":    ("Urine Sugar", "scale 0–5", "0"),
    "rbc":   ("Red Blood Cells", "urine", "normal"),
    "pc":    ("Pus Cells", "urine", "normal"),
    "pcc":   ("Pus Cell Clumps", "urine", "notpresent"),
    "ba":    ("Bacteria", "urine", "notpresent"),
    "bgr":   ("Random Blood Glucose", "mg/dL", "70–140"),
    "bu":    ("Blood Urea", "mg/dL", "15–40"),
    "sc":    ("Serum Creatinine", "mg/dL", "0.6–1.3"),
    "hemo":  ("Hemoglobin", "g/dL", "12.0–17.0"),
    "pcv":   ("Packed Cell Volume", "%", "36–50"),
    "rc":    ("RBC Count", "M/cmm", "4.2–5.9"),
    "htn":   ("Hypertension", "history", "no"),
    "dm":    ("Diabetes Mellitus", "history", "no"),
    "cad":   ("Coronary Artery Disease", "history", "no"),
    "appet": ("Appetite", "status", "good"),
    "pe":    ("Pedal Edema", "symptom", "no"),
    "ane":   ("Anemia", "symptom", "no"),
}

CATEGORICAL_MAP = {
    "rbc":   {"normal": 0, "abnormal": 1},
    "pc":    {"normal": 0, "abnormal": 1},
    "pcc":   {"notpresent": 0, "present": 1},
    "ba":    {"notpresent": 0, "present": 1},
    "htn":   {"no": 0, "yes": 1},
    "dm":    {"no": 0, "yes": 1},
    "cad":   {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe":    {"no": 0, "yes": 1},
    "ane":   {"no": 0, "yes": 1},
}

DEFAULTS = {
    "age": 50, "bp": 80, "al": 0, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 120, "bu": 40, "sc": 1.2, "hemo": 13.0,
    "pcv": 40, "rc": 5.0,
    "htn": "no", "dm": "no", "cad": "no",
    "appet": "good", "pe": "no", "ane": "no",
}


# ============================================================
# 3. CACHED MODEL & GROQ LLM CHAINS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_ml_model(model_path: str):
    if os.path.exists(model_path):
        return joblib.load(model_path), None
    if os.path.exists(FALLBACK_MODEL_PATH):
        return joblib.load(FALLBACK_MODEL_PATH), None
    return None, f"Model file not found at '{model_path}'."


@st.cache_resource(show_spinner=False)
def get_llm_chains(api_key: Optional[str] = None, model_name: str = "openai/gpt-oss-120b"):
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    llm = ChatGroq(model=model_name)
    parser = PydanticOutputParser(pydantic_object=Extractor)

    extract_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are a medical laboratory data extraction assistant for a kidney disease project.
Extract structured patient information from the text below.

Return values EXACTLY as follows:
- htn, dm, cad, pe, ane : "yes" or "no"
- rbc, pc               : "normal" or "abnormal"
- pcc, ba               : "present" or "notpresent"
- appet                 : "good" or "poor"
- numeric fields         : numbers only (no units)
- If a value is missing, leave it null.

{format_instructions}"""),
        ("user", "{paragraph}"),
    ])

    extract_chain = extract_template | llm

    # Summary Prompt: Normal, easy-to-read tone for Doctor + Detailed, warm explanation for Patient
    summary_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are a caring, experienced kidney specialist. Write a clear, easy-to-read clinical and patient summary based on the patient's lab values and kidney prediction.

Format your response using these EXACT 5 section headers:

**Abnormal Factors:**
- List up to 4 abnormal lab values with their exact numbers and normal reference context in simple bullets (or state "- All key kidney biomarkers are within normal healthy ranges." if normal).

**Top CKD Drivers:**
- One clear, normal sentence explaining the main clinical factors behind this result.

**Doctor:**
Write 3 clear sentences in a normal, natural doctor-to-colleague tone (easy to read, not overly stiff or cryptic). Summarize the patient's overall kidney filtration status, highlight the most important lab numbers (such as serum creatinine, blood urea, hemoglobin, albumin, and blood pressure), and state the recommended clinical next step.

**Patient:**
Write EXACTLY ONE short paragraph of about 5 lines (4 to 5 short sentences, roughly 60 to 80 words). Use very simple everyday words, as if talking to a worried family member with no medical background. Say in plain language what the results mean for the kidneys, mention one or two key numbers and what they mean in simple terms, and end with one reassuring, encouraging sentence. Avoid medical jargon; if you must use a medical word, explain it right away. Do not use bullet points in this section.

**Actions:**
- Give exactly 3 very short bullet points (maximum 12 words each) in simple words: see a doctor, drink water and eat light food, and watch blood pressure or sugar.

Rules:
- Keep the Doctor section in a normal, clear, natural tone.
- The Patient section must be ONE paragraph only, simple and warm, never longer than 5 sentences."""),
        ("user",
         """Data: {patient_data}
Prediction: {prediction_label} ({confidence}%)

Summarize now.""")
    ])

    summary_chain = summary_template | llm
    return extract_chain, parser, summary_chain


# ============================================================
# 4. CORE LOGIC & HTML FORMATTING HELPERS
# ============================================================
def sanitize_user_data(user_data: dict) -> dict:
    clean = user_data.copy()
    if not clean.get("name"):
        clean["name"] = "Anonymous Patient"
    for k, v in DEFAULTS.items():
        if clean.get(k) is None or clean.get(k) == "":
            clean[k] = v
    return clean


def predict(user_data: dict, ml_model):
    """Copy -> encode -> predict. Original not modified."""
    data = user_data.copy()

    for k, v in DEFAULTS.items():
        if data.get(k) is None:
            data[k] = v

    for col, mapping in CATEGORICAL_MAP.items():
        val = str(data[col]).strip().lower()
        data[col] = mapping.get(val, 0)

    X = pd.DataFrame([data])[FEATURES].values

    pred = int(ml_model.predict(X)[0])
    proba = ml_model.predict_proba(X)[0]
    confidence = float(proba.max() * 100)

    return pred, confidence


EXTRACT_FORMAT = (
    'Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape: '
    '{"patients": [ {"name": string or null, "age": number or null, "bp": number or null, '
    '"al": number or null, "su": number or null, "rbc": string or null, "pc": string or null, '
    '"pcc": string or null, "ba": string or null, "bgr": number or null, "bu": number or null, '
    '"sc": number or null, "hemo": number or null, "pcv": number or null, "rc": number or null, '
    '"htn": string or null, "dm": string or null, "cad": string or null, "appet": string or null, '
    '"pe": string or null, "ane": string or null} ]}. '
    'The text may describe ONE patient or SEVERAL patients: output exactly one object inside '
    '"patients" for each patient, in the order they appear. Use null for any missing value.'
)


def _loads_lenient(raw: str):
    """Parse JSON even if the model wrapped it in ``` fences or added extra words."""
    txt = (raw or "").strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE).strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", txt, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    raise ValueError("The AI did not return readable patient data. Please try again.")


def _clean_patient(d: dict) -> dict:
    """Normalise one extracted patient so the ML model always receives valid values."""
    out = {}
    for k in Extractor.model_fields:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a", "na", "unknown"):
            continue
        if k == "name":
            out[k] = str(v).strip()
        elif k in CATEGORICAL_MAP:
            s = re.sub(r"[\s_\-]", "", str(v).lower())
            if k in ("pcc", "ba"):
                out[k] = "present" if s in ("present", "yes", "true", "y", "positive", "seen") else "notpresent"
            elif k in ("rbc", "pc"):
                out[k] = "abnormal" if s.startswith("abn") or s in ("present", "yes", "positive") else "normal"
            elif k == "appet":
                out[k] = "poor" if s in ("poor", "bad", "low", "reduced") else "good"
            else:  # htn, dm, cad, pe, ane
                out[k] = "yes" if s in ("yes", "y", "true", "present", "positive") else "no"
        else:
            try:
                num = float(re.sub(r"[^0-9.\-]", "", str(v)))
                out[k] = int(num) if num.is_integer() else num
            except ValueError:
                continue
    return Extractor(**out).model_dump()


def parse_patients(raw: str) -> list:
    """Accepts {"patients":[...]}, a bare list, or a single object -> list of clean patient dicts."""
    data = _loads_lenient(raw)
    if isinstance(data, dict):
        if isinstance(data.get("patients"), list):
            items = data["patients"]
        elif any(k in data for k in Extractor.model_fields):
            items = [data]
        else:
            items = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    patients = []
    for it in items:
        if isinstance(it, dict):
            cleaned = _clean_patient(it)
            if any(cleaned.get(k) is not None for k in FEATURES):
                patients.append(cleaned)
    if not patients:
        raise ValueError("No patient lab values were found in the text.")
    return patients


def extract_from_text(text: str, api_key: Optional[str], model_name: str) -> list:
    """Returns a LIST of patient dicts (works for one patient or many in the same file)."""
    extract_chain, _, _ = get_llm_chains(api_key, model_name)
    result = extract_chain.invoke({
        "paragraph": text,
        "format_instructions": EXTRACT_FORMAT,
    })
    return parse_patients(result.content)


def generate_summary(user_data: dict, pred: int, confidence: float, api_key: Optional[str], model_name: str) -> str:
    _, _, summary_chain = get_llm_chains(api_key, model_name)
    label = "CKD (Chronic Kidney Disease)" if pred == 1 else "Healthy (No CKD)"

    response = summary_chain.invoke({
        "patient_data": json.dumps(user_data, indent=2),
        "prediction_label": label,
        "confidence": round(confidence, 2),
    })
    return response.content


def parse_summary_sections(summary_text: str) -> dict:
    sections = {
        "abnormal": "",
        "drivers": "",
        "doctor": "",
        "patient": "",
        "actions": "",
    }
    if not summary_text:
        return sections

    patterns = {
        "abnormal": r"\*\*Abnormal Factors:\*\*(.*?)(?=\*\*Top CKD Drivers:|\*\*Doctor:|\*\*Patient:|\*\*Actions:|$)",
        "drivers":  r"\*\*Top CKD Drivers:\*\*(.*?)(?=\*\*Doctor:|\*\*Patient:|\*\*Actions:|$)",
        "doctor":   r"\*\*Doctor:\*\*(.*?)(?=\*\*Patient:|\*\*Actions:|$)",
        "patient":  r"\*\*Patient:\*\*(.*?)(?=\*\*Actions:|$)",
        "actions":  r"\*\*Actions:\*\*(.*?)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, summary_text, re.DOTALL | re.IGNORECASE)
        if m:
            sections[key] = m.group(1).strip()

    if not sections["doctor"] and not sections["patient"]:
        sections["doctor"] = summary_text
        sections["patient"] = summary_text

    return sections


def md_to_clean_html(text: str) -> str:
    """Converts markdown bolding and bullet lists into clean HTML for rounded cards."""
    if not text:
        return "<p class='readable-paragraph'>No details available.</p>"

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    html_parts = []
    in_list = False

    for line in lines:
        escaped = html.escape(line)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

        if escaped.startswith("- ") or escaped.startswith("* ") or escaped.startswith("• "):
            if not in_list:
                html_parts.append("<ul class='readable-list'>")
                in_list = True
            item_content = escaped[2:].strip()
            html_parts.append(f"<li>{item_content}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p class='readable-paragraph'>{escaped}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts)


def flat_html(markup: str) -> str:
    """Collapse indentation/newlines so Streamlit's markdown never treats HTML as a code block."""
    return re.sub(r"\s*\n\s*", " ", markup).strip()


def load_all_records() -> list:
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, dict):
                return [records]
            return records if isinstance(records, list) else []
        except Exception:
            return []
    return []


def save_record(user_data: dict, pred: int, confidence: float, summary: str = "") -> dict:
    records = load_all_records()
    entry = {
        "id": len(records) + 1,
        "timestamp": datetime.now().isoformat(),
        "user_data": user_data.copy(),
        "prediction": {
            "label": "CKD" if pred == 1 else "Healthy",
            "value": pred,
            "confidence": round(confidence, 2),
        },
        "summary": summary,
    }
    records.append(entry)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return entry


def is_abnormal_biomarker(key: str, val) -> bool:
    try:
        if val is None:
            return False
        if key == "sc" and float(val) > 1.5:
            return True
        if key == "bu" and float(val) > 40:
            return True
        if key == "hemo" and float(val) < 12.0:
            return True
        if key == "al" and float(val) >= 1:
            return True
        if key == "su" and float(val) >= 1:
            return True
        if key == "bp" and float(val) >= 90:
            return True
        if key == "bgr" and float(val) > 140:
            return True
        if key in ("rbc", "pc") and str(val).lower() == "abnormal":
            return True
        if key in ("pcc", "ba") and str(val).lower() == "present":
            return True
        if key in ("htn", "dm", "cad", "pe", "ane") and str(val).lower() == "yes":
            return True
        if key == "appet" and str(val).lower() == "poor":
            return True
    except (ValueError, TypeError):
        pass
    return False


# ============================================================
# 5. SESSION STATE INITIALIZATION
# ============================================================
if "patient_form_data" not in st.session_state:
    st.session_state.patient_form_data = {"name": "", **DEFAULTS}

if "latest_entry" not in st.session_state:
    st.session_state.latest_entry = None

if "extracted_raw" not in st.session_state:
    st.session_state.extracted_raw = None

if "clinical_note_input" not in st.session_state:
    st.session_state.clinical_note_input = (
        "Patient Name: Ahmed Raza, Age 62. Presents with pedal edema and poor appetite. "
        "History of hypertension (htn: yes) and diabetes mellitus (dm: yes), no coronary artery disease. "
        "Lab findings: BP 100 mm/Hg, Albumin 3, Sugar 2, RBC abnormal, Pus cells abnormal, "
        "pus cell clumps present, bacteria notpresent. Random blood glucose 240 mg/dL, "
        "Blood Urea 86 mg/dL, Serum Creatinine 3.8 mg/dL, Hemoglobin 9.4 g/dL, PCV 29%, "
        "RBC count 3.4 million/cmm. Patient also has anemia."
    )


# ============================================================
# 6. SIDEBAR: CLEAN SYSTEM STATUS & SETTINGS
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 NephroCare AI")
    st.caption("Smart Kidney Disease Screening System")
    st.divider()

    ml_model, model_err = load_ml_model(DEFAULT_MODEL_PATH)
    env_groq_key = os.getenv("GROQ_API_KEY", "")

    if ml_model is not None:
        st.success("ML Kidney Model: Active", icon="✅")
    else:
        st.warning("ML Model Not Found", icon="⚠️")

    if env_groq_key:
        st.success("Groq AI Engine: Connected", icon="🧠")

    all_saved = load_all_records()
    st.metric("Total Patients Screened", len(all_saved))

    with st.expander("⚙️ Engine & API Settings", expanded=(ml_model is None or not env_groq_key)):
        model_path_input = st.text_input("Model Path (.pkl)", value=DEFAULT_MODEL_PATH)
        uploaded_pkl = st.file_uploader("Or upload kidney_model.pkl", type=["pkl", "joblib"])
        if uploaded_pkl is not None:
            with open(FALLBACK_MODEL_PATH, "wb") as f:
                f.write(uploaded_pkl.getbuffer())
            st.cache_resource.clear()
            ml_model, model_err = load_ml_model(model_path_input)

        llm_model_name = st.text_input("Groq LLM Model", value="openai/gpt-oss-120b")
        groq_api_key = st.text_input(
            "Groq API Key",
            value=env_groq_key,
            type="password",
            placeholder="gsk_...",
        )

    st.divider()
    st.caption("💡 **Exhibition Tip:** Use the 1-Click Demo Buttons on the main screen to test CKD vs. Healthy cases live.")


# ============================================================
# 7. TOP HERO BANNER
# ============================================================
st.markdown(
    """
    <div class="top-banner">
        <div>
            <h1>🩺 NephroCare AI — Kidney Disease Diagnostic Portal</h1>
            <p>AI Lab Note Extraction • Machine Learning CKD Prediction • Doctor & Patient Care Reports</p>
        </div>
        <div class="status-chip">🏥 Clinical Decision Support</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7b. LIVE "AI IS THINKING" PANEL + REGISTRY TABLE HELPERS
# ============================================================
# Pipeline flow: each stage is (kind, ref, icon, title, detail)
#   "send" = data travelling along connector `ref` (node ref -> node ref+1)
#   "work" = node `ref` is busy
AI_FLOW = {
    "nodes": [
        ("📄", "Patient Data", "Clinical note", "src"),
        ("🤖", "LLM", "Scanning", "llm"),
        ("🌳", "ML Model", "Ensemble trees", "ml"),
        ("🤖", "LLM", "Summary", "llm"),
        ("📋", "Report", "Doctor + Patient", "out"),
    ],
    "stages": [
        ("send", 0, "📤", "Sending patient data to the LLM",
         "Your clinical note is on its way to the AI"),
        ("work", 1, "🔬", "LLM is scanning the note",
         "Finding patient details and all 20 kidney biomarkers"),
        ("send", 1, "📦", "Sending extracted data to the ML model",
         "Passing the structured lab values on for prediction"),
        ("work", 2, "🌳", "Ensemble decision trees are voting",
         "Many decision trees analyse the values and predict CKD or healthy"),
        ("send", 2, "📨", "Sending the prediction back to the LLM",
         "The result and lab values go to the AI for explanation"),
        ("work", 3, "✍️", "LLM is writing the summaries",
         "Drafting the doctor summary and a simple patient guide"),
        ("send", 3, "📋", "Preparing your report",
         "Putting everything together"),
    ],
}

MANUAL_FLOW = {
    "nodes": [
        ("📝", "Lab Inputs", "20 markers", "src"),
        ("🌳", "ML Model", "Ensemble trees", "ml"),
        ("🤖", "LLM", "Summary", "llm"),
        ("📋", "Report", "Doctor + Patient", "out"),
    ],
    "stages": [
        ("send", 0, "📤", "Sending your 20 lab inputs to the ML model",
         "Every biomarker is checked and encoded"),
        ("work", 1, "🌳", "Ensemble decision trees are voting",
         "Many decision trees analyse the values and predict CKD or healthy"),
        ("send", 1, "📨", "Sending the prediction to the LLM",
         "The result and lab values go to the AI for explanation"),
        ("work", 2, "✍️", "LLM is writing the summaries",
         "Drafting the doctor summary and a simple patient guide"),
        ("send", 2, "📋", "Preparing your report",
         "Putting everything together"),
    ],
}

KEY_CHIP_FEATS = ["sc", "bu", "hemo", "bp", "bgr", "al"]


def key_found_chips(data: dict) -> list:
    """Key markers to pop up live in the panel once extraction finishes."""
    chips = []
    for k in KEY_CHIP_FEATS:
        v = data.get(k)
        if v is not None:
            chips.append((FEATURE_META[k][0], v, is_abnormal_biomarker(k, v)))
    return chips


def _flow_states(flow: dict, idx: int):
    nodes, stages = flow["nodes"], flow["stages"]
    n, last = len(stages), len(nodes) - 1
    all_done = idx >= n

    node_states = []
    for i in range(len(nodes)):
        if all_done:
            node_states.append("done")
        elif i == 0:
            node_states.append("active" if idx == 0 else "done")
        elif i == last:
            node_states.append("active" if idx == n - 1 else "todo")
        else:
            w = next(si for si, s in enumerate(stages) if s[0] == "work" and s[1] == i)
            node_states.append("done" if idx > w else ("active" if idx == w else "todo"))

    conn_states = []
    for j in range(len(nodes) - 1):
        sj = next(si for si, s in enumerate(stages) if s[0] == "send" and s[1] == j)
        conn_states.append("sending" if idx == sj else ("done" if idx > sj else "todo"))
    return node_states, conn_states


def show_thinking(slot, flow: dict, idx: int, found: Optional[list] = None):
    """Render the animated pipeline panel (data -> LLM -> ML trees -> LLM -> report) into `slot`."""
    nodes, stages = flow["nodes"], flow["stages"]
    n = len(stages)
    all_done = idx >= n
    if all_done:
        icon, title, detail, pct = "✅", "All done — your report is ready", "Check your summary and records", 100
    else:
        _, _, icon, title, detail = stages[idx]
        pct = int((idx + 0.5) / n * 100)

    node_states, conn_states = _flow_states(flow, idx)
    row = ""
    for i, (nicon, name, sub, kind) in enumerate(nodes):
        stt = node_states[i]
        extra = " scanning" if (kind == "llm" and stt == "active") else ""
        sub_html = html.escape(sub)
        if kind == "ml" and stt == "active":
            sub_html = '<span class="fl-trees"><b>🌳</b><b>🌲</b><b>🌳</b><b>🌲</b><b>🌳</b></span>'
        row += (
            f'<div class="fl-node {stt}{extra}"><div class="fl-ico">{nicon}</div>'
            f'<div class="fl-name">{html.escape(name)}</div><div class="fl-sub">{sub_html}</div></div>'
        )
        if i < len(nodes) - 1:
            row += f'<div class="fl-conn {conn_states[i]}"><i></i><i></i><i></i></div>'

    chips = ""
    if found:
        chips = '<div class="ai-chips">'
        for i, (lbl, val, flag) in enumerate(found):
            cls = "ai-chip ai-chip-flag" if flag else "ai-chip"
            mark = "⚠️" if flag else "✓"
            chips += (
                f'<span class="{cls}" style="animation-delay:{i * 0.1:.2f}s">'
                f'{mark} {html.escape(str(lbl))}: {html.escape(str(val))}</span>'
            )
        chips += "</div>"

    dots = "" if all_done else '<span class="ai-dots"><b></b><b></b><b></b></span>'
    core_cls = "ai-core ai-core-done mini" if all_done else "ai-core mini"
    done_cls = " ai-think-done" if all_done else ""

    markup = f"""
    <div class="ai-think flow-panel{done_cls}">
        <div class="flow-head">
            <div class="{core_cls}">{icon}</div>
            <div class="ai-body">
                <span class="pill-tag">✨ NephroCare AI pipeline</span>
                <h3 class="ai-title">{html.escape(title)}{dots}</h3>
                <p class="ai-detail">{html.escape(detail)}</p>
            </div>
            <div class="ai-pct big">{pct}%</div>
        </div>
        <div class="ai-bar-row"><div class="ai-bar"><div class="ai-bar-fill" style="width:{pct}%;"></div></div></div>
        <div class="fl-row">{row}</div>
        {chips}
    </div>
    """
    slot.markdown(flat_html(markup), unsafe_allow_html=True)


TABLE_LABELS = {
    "age": "Age", "bp": "BP", "al": "Albumin", "su": "Sugar", "rbc": "RBC",
    "pc": "Pus Cells", "pcc": "PC Clumps", "ba": "Bacteria", "bgr": "Glucose",
    "bu": "Urea", "sc": "Creatinine", "hemo": "Hemoglobin", "pcv": "PCV",
    "rc": "RBC Count", "htn": "Hypertension", "dm": "Diabetes", "cad": "CAD",
    "appet": "Appetite", "pe": "Pedal Edema", "ane": "Anemia",
}
NUMERIC_UNITS = {"years", "mm/Hg", "mg/dL", "g/dL", "%", "M/cmm"}


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def build_registry_table(recs: list) -> str:
    """Styled HTML table: report info + AI prediction + all 20 patient inputs."""
    grp = (
        '<tr class="grp">'
        '<th colspan="3" class="g-report">📄 Report</th>'
        '<th colspan="2" class="g-pred">🎯 AI Prediction</th>'
        f'<th colspan="{len(FEATURES)}" class="g-data">🧪 Patient Data (20 inputs)</th>'
        '</tr>'
    )
    cols = (
        '<tr class="cols">'
        '<th class="sticky-1">ID</th><th class="sticky-2">Patient</th><th>Date</th>'
        '<th>Diagnosis</th><th>Confidence</th>'
    )
    for f in FEATURES:
        label, unit, _ = FEATURE_META[f]
        small = f"<small>{html.escape(unit)}</small>" if unit in NUMERIC_UNITS else ""
        cols += f'<th title="{html.escape(label)}">{html.escape(TABLE_LABELS[f])}{small}</th>'
    cols += "</tr>"

    body = ""
    for r in recs:
        ud = sanitize_user_data(r.get("user_data", {}))
        pr = r.get("prediction", {})
        is_ckd = pr.get("value") == 1
        conf = float(pr.get("confidence", 0))
        ts = str(r.get("timestamp", ""))[:16].replace("T", " ")
        badge = (
            '<span class="tbl-badge tbl-badge-ckd">⚠️ CKD</span>' if is_ckd
            else '<span class="tbl-badge tbl-badge-ok">✅ Healthy</span>'
        )
        fill = "ckd" if is_ckd else "ok"
        body += (
            "<tr>"
            f'<td class="sticky-1 id-cell">#{html.escape(str(r.get("id", "")))}</td>'
            f'<td class="sticky-2 name-cell">{html.escape(str(ud.get("name") or "Anonymous Patient"))}</td>'
            f'<td class="date-cell">{ts}</td>'
            f"<td>{badge}</td>"
            f'<td><div class="conf-mini"><div class="conf-mini-track">'
            f'<div class="conf-mini-fill {fill}" style="width:{max(0.0, min(100.0, conf)):.1f}%;"></div></div>'
            f"<span>{conf:.1f}%</span></div></td>"
        )
        for f in FEATURES:
            v = ud.get(f, DEFAULTS[f])
            cls = ' class="cell-flag"' if is_abnormal_biomarker(f, v) else ""
            body += f"<td{cls}>{html.escape(_fmt_val(v))}</td>"
        body += "</tr>"

    return (
        '<div class="reg-wrap">'
        '<div class="reg-head"><div>'
        '<span class="pill-tag">🗂️ Patient Registry</span>'
        '<h3 class="card-title">History of Patient Data & AI Predictions</h3>'
        f'</div><div class="reg-count">{len(recs)} record{"s" if len(recs) != 1 else ""}</div></div>'
        '<div class="reg-scroll"><table class="reg-table">'
        f"<thead>{grp}{cols}</thead><tbody>{body}</tbody></table></div></div>"
    )


def registry_to_df(recs: list) -> pd.DataFrame:
    rows = []
    for r in recs:
        ud = sanitize_user_data(r.get("user_data", {}))
        pr = r.get("prediction", {})
        row = {
            "Report ID": r.get("id"),
            "Date": str(r.get("timestamp", ""))[:16].replace("T", " "),
            "Patient": ud.get("name"),
            "Prediction": pr.get("label"),
            "Confidence %": pr.get("confidence"),
        }
        for f in FEATURES:
            row[FEATURE_META[f][0]] = ud.get(f, DEFAULTS[f])
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 8. RENDER DIAGNOSTIC RESULTS (PREMIUM SUMMARY STAGE)
# ============================================================
def render_diagnostic_report(entry: dict, key_prefix: str = "t1"):
    pred_info = entry["prediction"]
    is_ckd = pred_info["value"] == 1
    conf = float(pred_info["confidence"])
    u_data = sanitize_user_data(entry["user_data"])
    patient_name = html.escape(str(u_data.get("name") or "Anonymous Patient"))
    patient_age = html.escape(str(u_data.get("age", "—")))
    timestamp = entry["timestamp"][:16].replace("T", " ")
    parsed = parse_summary_sections(entry.get("summary", ""))

    flagged_n = sum(
        1 for f in FEATURES if is_abnormal_biomarker(f, u_data.get(f, DEFAULTS[f]))
    )

    theme = "ckd" if is_ckd else "ok"
    accent = "#E11D48" if is_ckd else "#059669"
    ring_deg = max(0.0, min(100.0, conf)) * 3.6

    if is_ckd:
        risk_text = "⚠️ High Risk • Nephrology Attention Needed"
        headline = "Chronic Kidney Disease (CKD) Detected"
    else:
        risk_text = "✅ Low Risk • Normal Renal Function"
        headline = "Healthy Kidney Profile (No CKD Detected)"

    flag_chip_cls = "meta-chip meta-chip-warn" if flagged_n else "meta-chip meta-chip-good"

    doctor_html = md_to_clean_html(parsed["doctor"] or entry.get("summary", ""))
    drivers_html = md_to_clean_html(parsed["drivers"]) if parsed["drivers"] else ""
    abnormal_html = md_to_clean_html(parsed["abnormal"]) if parsed["abnormal"] else ""
    patient_html = (
        '<p class="readable-paragraph patient-para">'
        + html.escape(to_single_paragraph(parsed["patient"] or entry.get("summary", "")))
        + "</p>"
    )
    actions_html = md_to_clean_html(parsed["actions"]) if parsed["actions"] else ""

    # ---- Hero strip ----
    hero = f"""
    <div class="sum-hero sum-hero-{theme}">
        <div class="conf-ring" style="background: conic-gradient({accent} {ring_deg:.1f}deg, #E2E8F0 0deg);">
            <div class="conf-ring-inner">
                <strong style="color:{accent};">{conf:.1f}%</strong>
                <span>Confidence</span>
            </div>
        </div>
        <div class="sum-hero-text">
            <span class="risk-chip risk-chip-{theme}">{risk_text}</span>
            <h2>{headline}</h2>
            <div class="meta-row">
                <span class="meta-chip">👤 {patient_name} · {patient_age} yrs</span>
                <span class="meta-chip">🆔 Report #{entry['id']}</span>
                <span class="meta-chip">🕒 {timestamp}</span>
                <span class="{flag_chip_cls}">🧪 {flagged_n} of 20 markers flagged</span>
            </div>
        </div>
    </div>
    """

    # ---- Doctor card ----
    doc_card = f"""
    <div class="sum-card sum-card-doc">
        <div class="sum-card-head">
            <div class="sum-icon sum-icon-doc">👨‍⚕️</div>
            <div>
                <span class="pill-tag">Doctor's Side • Clinical Overview</span>
                <h3 class="card-title">Physician Assessment Summary</h3>
            </div>
        </div>
        <p class="card-sub">Clear, natural clinical summary &amp; key diagnostic findings</p>
        <div class="sum-body">{doctor_html}</div>
    """
    if drivers_html:
        doc_card += f"""
        <div class="sum-section">
            <div class="sum-section-title" style="color:#0369A1;">🎯 Main Clinical Drivers</div>
            {drivers_html}
        </div>
        """
    if abnormal_html:
        doc_card += f"""
        <div class="sum-section sum-section-warn">
            <div class="sum-section-title" style="color:#BE123C;">🔬 Abnormal Biomarkers Flagged</div>
            {abnormal_html}
        </div>
        """
    doc_card += "</div>"

    # ---- Patient card ----
    pat_card = f"""
    <div class="sum-card sum-card-pat">
        <div class="sum-card-head">
            <div class="sum-icon sum-icon-pat">💙</div>
            <div>
                <span class="pill-tag" style="background:#CCFBF1; color:#0F766E;">Patient's Side • Health Guide</span>
                <h3 class="card-title">What Your Results Mean for You</h3>
            </div>
        </div>
        <p class="card-sub">Easy-to-understand explanation &amp; daily kidney care plan</p>
        <div class="sum-body">{patient_html}</div>
    """
    if actions_html:
        pat_card += f"""
        <div class="sum-section sum-section-teal">
            <div class="sum-section-title" style="color:#0F766E;">✅ Recommended Next Steps &amp; Daily Care Plan</div>
            {actions_html}
        </div>
        """
    pat_card += "</div>"

    st.markdown(
        flat_html(f'<div class="summary-stage">{hero}<div class="sum-grid">{doc_card}{pat_card}</div></div>'),
        unsafe_allow_html=True,
    )

    # ---- Complete 20-biomarker panel ----
    tiles_html = '<div class="bio-panel-box">'
    tiles_html += (
        '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">'
        '<div>'
        '<span class="pill-tag">🧪 Complete Lab Panel</span>'
        '<h3 class="card-title">All 20 Kidney Biomarkers & Reference Status</h3>'
        '</div>'
        '</div>'
        '<div class="bio-grid-20">'
    )
    for feat in FEATURES:
        label, unit, ref = FEATURE_META[feat]
        val = html.escape(str(u_data.get(feat, DEFAULTS[feat])))
        flagged = is_abnormal_biomarker(feat, u_data.get(feat, DEFAULTS[feat]))
        tile_cls = "bio-tile-flag" if flagged else "bio-tile-ok"
        badge_cls = "badge-flag" if flagged else "badge-ok"
        badge_txt = "⚠️ Abnormal" if flagged else "✓ Normal"
        val_color = "#BE123C" if flagged else "#0F172A"

        tiles_html += (
            f'<div class="{tile_cls}">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">'
            f'<span style="font-size:0.76rem; font-weight:700; color:#475569;">{label}</span>'
            f'<span class="{badge_cls}">{badge_txt}</span>'
            f'</div>'
            f'<div style="font-size:1.18rem; font-weight:800; color:{val_color};">{val} '
            f'<span style="font-size:0.75rem; font-weight:500; color:#64748B;">{unit}</span></div>'
            f'<div style="font-size:0.72rem; color:#94A3B8; margin-top:0.2rem;">Normal Ref: {ref}</div>'
            f'</div>'
        )
    tiles_html += "</div></div>"
    st.markdown(tiles_html, unsafe_allow_html=True)

    # ---- Download buttons ----
    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "⬇️ Download Patient Record (JSON)",
            data=json.dumps(entry, indent=2),
            file_name=f"patient_report_{entry['id']}.json",
            mime="application/json",
            use_container_width=True,
            key=f"dl_json_{entry['id']}_{key_prefix}",
        )
    with dcol2:
        md_report = (
            f"# NephroCare AI — Kidney Diagnostic Report #{entry['id']}\n"
            f"**Patient:** {u_data.get('name')} (Age: {u_data.get('age')})\n"
            f"**Diagnosis:** {pred_info['label']} ({conf:.2f}% Confidence)\n\n"
            f"{entry.get('summary', '')}\n"
        )
        st.download_button(
            "📄 Download Printable Summary (.md)",
            data=md_report,
            file_name=f"kidney_report_{entry['id']}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_md_{entry['id']}_{key_prefix}",
        )


# ============================================================
# 8b. POPUP SCREENING FLOW + PATIENT SCREEN
# ============================================================
if "view" not in st.session_state:
    st.session_state.view = "home"


def to_single_paragraph(text: str, max_sentences: int = 5) -> str:
    """Flatten the patient section into ONE simple paragraph of at most 5 sentences."""
    if not text:
        return "No details available."
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    t = re.sub(r"^\s*[-*•]\s*", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    sentences = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(sentences[:max_sentences]).strip()


def _screening_cta():
    st.markdown(
        '<div class="cta-note">✅ Analysis complete! Check your summary and records.</div>',
        unsafe_allow_html=True,
    )
    if st.button("📋 Check your summary & records  →", type="primary",
                 use_container_width=True, key="cta_go_patient"):
        st.session_state.view = "patient"
        st.rerun()


SEND_PAUSE = 0.3   # seconds a "data is travelling" step stays visible
ML_PAUSE = 0.5     # seconds the decision-tree voting animation stays visible


def _make_summaries(patients: list, preds: list, api_key, model_name) -> list:
    """Write one summary per patient (in parallel when there are several)."""
    _, _, chain = get_llm_chains(api_key, model_name)  # warm the cache on the main thread

    def one(pair):
        p, (pred, conf) = pair
        label = "CKD (Chronic Kidney Disease)" if pred == 1 else "Healthy (No CKD)"
        try:
            return chain.invoke({
                "patient_data": json.dumps(p, indent=2),
                "prediction_label": label,
                "confidence": round(conf, 2),
            }).content
        except Exception as e:
            return ("**Doctor:** Clinical evaluation complete.\n\n"
                    f"**Patient:** The AI summary could not be written right now ({e}).")

    pairs = list(zip(patients, preds))
    if len(pairs) == 1:
        return [one(pairs[0])]
    with ThreadPoolExecutor(max_workers=min(4, len(pairs))) as pool:
        return list(pool.map(one, pairs))


@st.dialog("🩺 NephroCare AI — Scanning Patient Data", width="large")
def run_screening_dialog(mode: str, data, ml_model, api_key, model_name, job_id: str):
    """Popup that runs the whole pipeline with a live flow animation, then offers the redirect."""
    flow = AI_FLOW if mode == "ai" else MANUAL_FLOW
    n = len(flow["stages"])
    done_key = f"job_done_{job_id}"
    slot = st.empty()

    # A click inside the popup re-runs this function: don't re-run the AI, just show the finish state.
    if st.session_state.get(done_key):
        show_thinking(slot, flow, n, found=st.session_state.get("last_chips"))
        _screening_cta()
        return

    step = "Extraction" if mode == "ai" else "Prediction"
    try:
        if mode == "ai":
            show_thinking(slot, flow, 0)                       # data -> LLM
            time.sleep(SEND_PAUSE)

            show_thinking(slot, flow, 1)                       # LLM scans the note
            patients = extract_from_text(data, api_key, model_name)
            chips = key_found_chips(patients[0])
            if len(patients) > 1:
                chips.insert(0, ("Patients found", len(patients), False))
            st.session_state.extracted_raw = patients[0]
            st.session_state.patient_form_data = sanitize_user_data(patients[0])

            show_thinking(slot, flow, 2, found=chips)          # LLM -> ML model
            time.sleep(SEND_PAUSE)

            step = "Prediction"
            show_thinking(slot, flow, 3, found=chips)          # ensemble trees vote
            preds = [predict(p, ml_model) for p in patients]
            time.sleep(ML_PAUSE)

            show_thinking(slot, flow, 4, found=chips)          # ML -> LLM
            time.sleep(SEND_PAUSE)

            step = "Summary"
            show_thinking(slot, flow, 5, found=chips)          # LLM writes summaries
            summaries = _make_summaries(patients, preds, api_key, model_name)

            show_thinking(slot, flow, 6, found=chips)          # build report
            entries = [
                save_record(p, pr, cf, sm)
                for p, (pr, cf), sm in zip(patients, preds, summaries)
            ]
        else:
            chips = key_found_chips(data)
            show_thinking(slot, flow, 0, found=chips)          # inputs -> ML model
            time.sleep(SEND_PAUSE)

            show_thinking(slot, flow, 1, found=chips)          # ensemble trees vote
            pred, confidence = predict(data, ml_model)
            time.sleep(ML_PAUSE)

            show_thinking(slot, flow, 2, found=chips)          # ML -> LLM
            time.sleep(SEND_PAUSE)

            step = "Summary"
            show_thinking(slot, flow, 3, found=chips)          # LLM writes summaries
            if api_key:
                summary = _make_summaries([data], [(pred, confidence)], api_key, model_name)[0]
            else:
                summary = ("**Doctor:** Groq API key not configured.\n\n"
                           "**Patient:** Please add your Groq API key to generate the patient guide.")

            show_thinking(slot, flow, 4, found=chips)          # build report
            entries = [save_record(data, pred, confidence, summary)]

        st.session_state.latest_batch = entries
        st.session_state.latest_entry = entries[0]
        st.session_state.last_chips = chips
    except Exception as e:
        slot.empty()
        st.error(f"{step} Error: {e}")
        return

    st.session_state[done_key] = True
    show_thinking(slot, flow, n, found=chips)
    st.toast(
        f"{len(entries)} report{'s' if len(entries) != 1 else ''} saved!", icon="✅"
    )
    _screening_cta()


def render_patient_screen():
    """Dedicated patient screen: summary + saved records (supports several patients from one upload)."""
    top1, _ = st.columns([1, 3])
    with top1:
        if st.button("←  New screening", use_container_width=True, key="back_home"):
            st.session_state.view = "home"
            st.rerun()

    batch = st.session_state.get("latest_batch") or (
        [st.session_state.latest_entry] if st.session_state.latest_entry else []
    )
    if not batch:
        st.info("No report yet. Run a screening first.")
        return

    entry = batch[0]
    if len(batch) > 1:
        st.markdown(
            f'<div class="pt-hero"><span class="pill-tag">👥 {len(batch)} patients found in this upload</span>'
            '<p style="margin-top:0.4rem;">Pick a patient to see their summary.</p></div>',
            unsafe_allow_html=True,
        )
        options = list(range(len(batch)))

        def _lbl(i):
            b = batch[i]
            nm = str(b.get("user_data", {}).get("name") or "Anonymous Patient")
            return f"{nm} — {b['prediction']['label']} ({b['prediction']['confidence']:.1f}%)"

        pick = st.radio("Patients in this upload", options, format_func=_lbl,
                        horizontal=True, key="batch_pick", label_visibility="collapsed")
        entry = batch[pick]

    name = str(entry.get("user_data", {}).get("name") or "Anonymous Patient")
    st.markdown(
        flat_html(
            f"""
            <div class="pt-hero">
                <span class="pill-tag">👤 Patient Screen</span>
                <h2>{html.escape(name)} — Summary &amp; Records</h2>
                <p>Your latest kidney screening result, an easy summary, and your saved history.</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    render_diagnostic_report(entry, key_prefix=f"pat{entry['id']}")

    st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)
    if len(batch) > 1:
        records_to_show = list(reversed(batch))
        caption = f"All {len(batch)} patients from this upload"
    else:
        records_to_show = [
            r for r in reversed(load_all_records())
            if str(r.get("user_data", {}).get("name") or "Anonymous Patient") == name
        ]
        caption = f"Saved records for {name}"
    if records_to_show:
        st.markdown(flat_html(build_registry_table(records_to_show)), unsafe_allow_html=True)
        st.caption(f"{caption} • Red cells = abnormal values • Scroll sideways to see all 20 inputs")


if st.session_state.view == "patient":
    render_patient_screen()
    st.stop()


# ============================================================
# 9. MAIN NAVIGATION TABS
# ============================================================
tab_intake, tab_manual, tab_history = st.tabs([
    "📋 1. AI Lab Note Screening (Paste / Upload)",
    "🎛️ 2. Manual Lab Calculator (20 Markers)",
    "🗂️ 3. Saved Patient Reports",
])


# ------------------------------------------------------------
# TAB 1: AI LAB NOTE SCREENING
# ------------------------------------------------------------
with tab_intake:
    col_in, col_snap = st.columns([1.1, 0.9], gap="large")

    with col_in:
        with st.container(border=True):
            st.markdown('<span class="pill-tag">Step 1 • Patient Input</span>', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title">📥 Enter or Upload Patient Lab Information</h3>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Choose a demo patient below, paste a clinical paragraph, or upload a lab text file.</p>', unsafe_allow_html=True)

            intake_mode = st.radio(
                "Input Method",
                options=["📝 Paste Clinical Note / Lab Text", "📂 Upload Lab Report File (.txt, .json, .csv)"],
                horizontal=True,
                label_visibility="collapsed",
            )

            raw_clinical_text = ""

            if intake_mode.startswith("📝"):
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔴 Load CKD Patient Demo (Ahmed, 62)", use_container_width=True):
                        st.session_state.clinical_note_input = (
                            "Patient Name: Ahmed Raza, Age 62. Presents with pedal edema and poor appetite. "
                            "History of hypertension (htn: yes) and diabetes mellitus (dm: yes), no coronary artery disease. "
                            "Lab findings: BP 100 mm/Hg, Albumin 3, Sugar 2, RBC abnormal, Pus cells abnormal, "
                            "pus cell clumps present, bacteria notpresent. Random blood glucose 240 mg/dL, "
                            "Blood Urea 86 mg/dL, Serum Creatinine 3.8 mg/dL, Hemoglobin 9.4 g/dL, PCV 29%, "
                            "RBC count 3.4 million/cmm. Patient also has anemia."
                        )
                with b2:
                    if st.button("🟢 Load Healthy Patient Demo (Sara, 34)", use_container_width=True):
                        st.session_state.clinical_note_input = (
                            "Patient Name: Sara Khan, 34 years old. Routine wellness checkup. Blood pressure 76 mm/Hg, "
                            "albumin 0, sugar 0, red blood cells normal, pus cells normal, no clumps or bacteria. "
                            "Random blood glucose 98 mg/dL, blood urea 24 mg/dL, serum creatinine 0.8 mg/dL, "
                            "hemoglobin 14.6 g/dL, PCV 44%, RBC count 5.1 million/cmm. No hypertension, "
                            "no diabetes, no coronary artery disease, good appetite, no pedal edema or anemia."
                        )

                raw_clinical_text = st.text_area(
                    "Patient Lab Note / Clinical Paragraph",
                    key="clinical_note_input",
                    height=180,
                    placeholder="Type or paste patient lab findings here...",
                )
            else:
                uploaded_report = st.file_uploader(
                    "Upload Patient Report File (.txt, .csv, .json)",
                    type=["txt", "md", "csv", "json"],
                )
                if uploaded_report is not None:
                    raw_clinical_text = uploaded_report.read().decode("utf-8", errors="ignore")
                    st.text_area("Uploaded File Preview", value=raw_clinical_text, height=150, disabled=True)

            run_btn = st.button(
                "🩺 Run Kidney Screening & Generate Doctor + Patient Report",
                type="primary",
                use_container_width=True,
            )

    if run_btn:
        if not raw_clinical_text.strip():
            st.error("Please enter a clinical note or upload a file first.")
        elif not groq_api_key:
            st.error("Please enter your GROQ_API_KEY in `.env` or the sidebar settings.")
        elif ml_model is None:
            st.error("ML Model (`kidney_model.pkl`) is not loaded. Check the sidebar settings.")
        else:
            # Opens the scanning popup at the top of the screen
            run_screening_dialog("ai", raw_clinical_text, ml_model, groq_api_key, llm_model_name, str(time.time()))

    # RIGHT COLUMN: SELF-CONTAINED ROUNDED SNAPSHOT CARD
    with col_snap:
        if st.session_state.extracted_raw is not None:
            ext = st.session_state.extracted_raw
            clean_ext = sanitize_user_data(ext)
            found_count = sum(1 for k in FEATURES if ext.get(k) is not None)
            p_name = html.escape(str(clean_ext.get("name") or "Anonymous Patient"))
            p_age = html.escape(str(clean_ext.get("age", "—")))

            key_feats = ["sc", "bu", "hemo", "bp", "bgr", "al", "htn", "dm"]
            pills_html = '<div class="lab-mini-grid">'
            for k in key_feats:
                lbl, unit, _ = FEATURE_META[k]
                val = html.escape(str(clean_ext.get(k)))
                flag = is_abnormal_biomarker(k, clean_ext.get(k))
                p_cls = "lab-pill-abnormal" if flag else "lab-pill-normal"
                v_col = "#BE123C" if flag else "#0F172A"
                icon = "⚠️ " if flag else "✓ "
                u_str = unit if unit not in ("history", "scale 0–5") else ""
                pills_html += (
                    f'<div class="{p_cls}">'
                    f'<span style="font-size:0.82rem; font-weight:700; color:#475569;">{lbl}</span>'
                    f'<span style="font-size:0.95rem; font-weight:800; color:{v_col};">{icon}{val} '
                    f'<small style="font-weight:500; color:#64748B;">{u_str}</small></span>'
                    f'</div>'
                )
            pills_html += "</div>"

            snap_card = f"""
            <div class="snapshot-card">
                <span class="pill-tag">Step 2 • AI Extracted Profile</span>
                <h3 class="card-title">🔬 Extracted Patient & Key Renal Markers</h3>
                <p class="card-sub">Automatically structured from the clinical note for ML classification.</p>
                <div class="patient-banner">
                    <div>
                        <div style="font-size:0.72rem; font-weight:800; color:#0369A1; text-transform:uppercase;">Patient Name & Age</div>
                        <div style="font-size:1.2rem; font-weight:800; color:#0F172A;">👤 {p_name} <span style="font-size:0.95rem; font-weight:600; color:#475569;">({p_age} yrs)</span></div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.72rem; font-weight:800; color:#0369A1; text-transform:uppercase;">Markers Extracted</div>
                        <div style="font-size:1.1rem; font-weight:800; color:#0284C7;">{found_count} / 20 Found</div>
                    </div>
                </div>
                {pills_html}
            </div>
            """
            st.markdown(flat_html(snap_card), unsafe_allow_html=True)
        else:
            st.markdown(
                flat_html(
                    """
                    <div class="snapshot-card">
                        <span class="pill-tag">Step 2 • AI Extracted Profile</span>
                        <h3 class="card-title">🔬 Extracted Patient & Key Renal Markers</h3>
                        <p class="card-sub">Your structured patient profile and highlighted lab values will appear here as soon as you click <strong>Run Kidney Screening</strong>.</p>
                        <div style="background:#F0F9FF; border:2px dashed #BAE6FD; border-radius:16px; padding:2.2rem 1.5rem; text-align:center; color:#0369A1; font-weight:600;">
                            👈 Select a Demo Patient or paste a lab note on the left, then click <strong>Run Kidney Screening</strong>.
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    if st.session_state.latest_entry is not None:
        render_diagnostic_report(st.session_state.latest_entry, key_prefix="t1")


# ------------------------------------------------------------
# TAB 2: MANUAL LAB CALCULATOR
# ------------------------------------------------------------
with tab_manual:
    with st.container(border=True):
        st.markdown('<span class="pill-tag">Interactive Lab Calculator</span>', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">🎛️ Enter or Edit All 20 Kidney Biomarkers</h3>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Modify any value below to test how specific lab markers affect the CKD prediction and patient guidance.</p>', unsafe_allow_html=True)

        cur = st.session_state.patient_form_data

        with st.form("manual_clinical_form"):
            st.markdown("##### 👤 Patient Details & Vitals")
            d1, d2, d3 = st.columns(3)
            with d1:
                f_name = st.text_input("Patient Full Name", value=str(cur.get("name") or ""))
            with d2:
                f_age = st.number_input("Age (years)", min_value=1, max_value=120, value=int(float(cur.get("age", 50))))
            with d3:
                f_bp = st.number_input("Blood Pressure (mm/Hg)", min_value=40, max_value=240, value=int(float(cur.get("bp", 80))))

            st.markdown("##### 🩸 Blood Chemistry & Hematology")
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            with b1:
                f_sc = st.number_input("Serum Creatinine (sc)", min_value=0.1, max_value=50.0, value=float(cur.get("sc", 1.2)), step=0.1)
            with b2:
                f_bu = st.number_input("Blood Urea (bu)", min_value=1.0, max_value=400.0, value=float(cur.get("bu", 40.0)), step=1.0)
            with b3:
                f_hemo = st.number_input("Hemoglobin (hemo)", min_value=3.0, max_value=22.0, value=float(cur.get("hemo", 13.0)), step=0.1)
            with b4:
                f_bgr = st.number_input("Blood Glucose (bgr)", min_value=20.0, max_value=600.0, value=float(cur.get("bgr", 120.0)), step=1.0)
            with b5:
                f_pcv = st.number_input("Packed Cell Vol (pcv)", min_value=10.0, max_value=70.0, value=float(cur.get("pcv", 40.0)), step=1.0)
            with b6:
                f_rc = st.number_input("RBC Count (rc)", min_value=1.5, max_value=9.0, value=float(cur.get("rc", 5.0)), step=0.1)

            st.markdown("##### 🧪 Urinalysis Markers")
            u1, u2, u3, u4, u5, u6 = st.columns(6)
            with u1:
                f_al = st.selectbox("Albumin (0–5)", options=[0, 1, 2, 3, 4, 5], index=int(float(cur.get("al", 0))))
            with u2:
                f_su = st.selectbox("Sugar (0–5)", options=[0, 1, 2, 3, 4, 5], index=int(float(cur.get("su", 0))))
            with u3:
                rbc_opts = ["normal", "abnormal"]
                f_rbc = st.selectbox("Red Blood Cells", options=rbc_opts, index=rbc_opts.index(str(cur.get("rbc", "normal")).lower() if str(cur.get("rbc", "normal")).lower() in rbc_opts else "normal"))
            with u4:
                pc_opts = ["normal", "abnormal"]
                f_pc = st.selectbox("Pus Cells", options=pc_opts, index=pc_opts.index(str(cur.get("pc", "normal")).lower() if str(cur.get("pc", "normal")).lower() in pc_opts else "normal"))
            with u5:
                pcc_opts = ["notpresent", "present"]
                f_pcc = st.selectbox("Pus Cell Clumps", options=pcc_opts, index=pcc_opts.index(str(cur.get("pcc", "notpresent")).lower() if str(cur.get("pcc", "notpresent")).lower() in pcc_opts else "notpresent"))
            with u6:
                ba_opts = ["notpresent", "present"]
                f_ba = st.selectbox("Bacteria", options=ba_opts, index=ba_opts.index(str(cur.get("ba", "notpresent")).lower() if str(cur.get("ba", "notpresent")).lower() in ba_opts else "notpresent"))

            st.markdown("##### 🩺 Medical History & Symptoms")
            h1, h2, h3, h4, h5, h6 = st.columns(6)
            yn_opts = ["no", "yes"]
            ap_opts = ["good", "poor"]
            with h1:
                f_htn = st.selectbox("Hypertension", options=yn_opts, index=yn_opts.index(str(cur.get("htn", "no")).lower() if str(cur.get("htn", "no")).lower() in yn_opts else "no"))
            with h2:
                f_dm = st.selectbox("Diabetes Mellitus", options=yn_opts, index=yn_opts.index(str(cur.get("dm", "no")).lower() if str(cur.get("dm", "no")).lower() in yn_opts else "no"))
            with h3:
                f_cad = st.selectbox("Coronary Artery Dis.", options=yn_opts, index=yn_opts.index(str(cur.get("cad", "no")).lower() if str(cur.get("cad", "no")).lower() in yn_opts else "no"))
            with h4:
                f_appet = st.selectbox("Appetite", options=ap_opts, index=ap_opts.index(str(cur.get("appet", "good")).lower() if str(cur.get("appet", "good")).lower() in ap_opts else "good"))
            with h5:
                f_pe = st.selectbox("Pedal Edema", options=yn_opts, index=yn_opts.index(str(cur.get("pe", "no")).lower() if str(cur.get("pe", "no")).lower() in yn_opts else "no"))
            with h6:
                f_ane = st.selectbox("Anemia", options=yn_opts, index=yn_opts.index(str(cur.get("ane", "no")).lower() if str(cur.get("ane", "no")).lower() in yn_opts else "no"))

            submit_manual = st.form_submit_button(
                "🩺 Calculate CKD Risk & Generate Doctor + Patient Report",
                type="primary",
                use_container_width=True,
            )

    if submit_manual:
        manual_payload = {
            "name": f_name.strip() or "Anonymous Patient",
            "age": f_age, "bp": f_bp, "al": f_al, "su": f_su,
            "rbc": f_rbc, "pc": f_pc, "pcc": f_pcc, "ba": f_ba,
            "bgr": f_bgr, "bu": f_bu, "sc": f_sc, "hemo": f_hemo,
            "pcv": f_pcv, "rc": f_rc, "htn": f_htn, "dm": f_dm,
            "cad": f_cad, "appet": f_appet, "pe": f_pe, "ane": f_ane,
        }
        st.session_state.patient_form_data = manual_payload
        st.session_state.extracted_raw = manual_payload

        if ml_model is None:
            st.error("ML Model (`kidney_model.pkl`) is not loaded.")
        else:
            # Opens the scanning popup at the top of the screen
            run_screening_dialog("manual", manual_payload, ml_model, groq_api_key, llm_model_name, str(time.time()))

    if st.session_state.latest_entry is not None:
        render_diagnostic_report(st.session_state.latest_entry, key_prefix="t2")


# ------------------------------------------------------------
# TAB 3: SAVED PATIENT REPORTS
# ------------------------------------------------------------
with tab_history:
    records = load_all_records()

    if not records:
        st.markdown(
            flat_html(
                """
                <div class="snapshot-card" style="text-align:center; padding:2.5rem;">
                    <h3 class="card-title">No Saved Patient Reports Yet</h3>
                    <p class="card-sub">Run a screening in Tab 1 or Tab 2 to automatically save patient records to <code>patient_reports.json</code>.</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        total_cases = len(records)
        ckd_cases = sum(1 for r in records if r.get("prediction", {}).get("value") == 1)
        healthy_cases = total_cases - ckd_cases
        avg_conf = sum(r.get("prediction", {}).get("confidence", 0) for r in records) / total_cases

        st.markdown(
            flat_html(
                f"""
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:1rem; margin-bottom:1.2rem;">
                    <div class="snapshot-card" style="padding:1.2rem 1.4rem;">
                        <div style="font-size:0.75rem; font-weight:800; color:#64748B; text-transform:uppercase;">Total Screenings</div>
                        <div style="font-size:2rem; font-weight:800; color:#0F172A;">{total_cases}</div>
                    </div>
                    <div class="snapshot-card" style="padding:1.2rem 1.4rem; border-color:#FDA4AF;">
                        <div style="font-size:0.75rem; font-weight:800; color:#9F1239; text-transform:uppercase;">CKD Detected</div>
                        <div style="font-size:2rem; font-weight:800; color:#BE123C;">{ckd_cases}</div>
                    </div>
                    <div class="snapshot-card" style="padding:1.2rem 1.4rem; border-color:#6EE7B7;">
                        <div style="font-size:0.75rem; font-weight:800; color:#065F46; text-transform:uppercase;">Healthy Profiles</div>
                        <div style="font-size:2rem; font-weight:800; color:#059669;">{healthy_cases}</div>
                    </div>
                    <div class="snapshot-card" style="padding:1.2rem 1.4rem; border-color:#BAE6FD;">
                        <div style="font-size:0.75rem; font-weight:800; color:#0369A1; text-transform:uppercase;">Avg Confidence</div>
                        <div style="font-size:2rem; font-weight:800; color:#0284C7;">{avg_conf:.1f}%</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        fc1, fc2, fc3, fc4 = st.columns([1.5, 1, 1, 1])
        with fc1:
            search_q = st.text_input("🔍 Search by Patient Name or Report ID", placeholder="Type patient name or ID...")
        with fc2:
            status_filter = st.selectbox("Filter by Diagnosis", ["All", "CKD", "Healthy"])

        filtered = []
        for rec in reversed(records):
            p_name = str(rec.get("user_data", {}).get("name") or "Anonymous Patient")
            p_label = rec.get("prediction", {}).get("label", "")
            r_id = str(rec.get("id", ""))
            if status_filter != "All" and p_label != status_filter:
                continue
            if search_q and (search_q.lower() not in p_name.lower() and search_q not in r_id):
                continue
            filtered.append(rec)

        with fc3:
            st.write("")
            st.write("")
            st.download_button(
                "⬇️ Export Table (CSV)",
                data=registry_to_df(filtered).to_csv(index=False),
                file_name="patient_registry.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=not filtered,
            )
        with fc4:
            st.write("")
            st.write("")
            st.download_button(
                "⬇️ Export Full Registry (JSON)",
                data=json.dumps(records, indent=2),
                file_name="patient_reports_export.json",
                mime="application/json",
                use_container_width=True,
            )

        if not filtered:
            st.info("No saved reports match your search or filter.")
        else:
            st.markdown(flat_html(build_registry_table(filtered)), unsafe_allow_html=True)
            st.caption(
                f"Showing {len(filtered)} of {total_cases} reports • Red cells = abnormal values • "
                "Scroll sideways to see all 20 patient inputs"
            )

            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
            label_map = {}
            for r in filtered:
                nm = str(r.get("user_data", {}).get("name") or "Anonymous Patient")
                pr = r.get("prediction", {})
                label_map[r.get("id")] = (
                    f"#{r.get('id')} — {nm} • {pr.get('label', '')} ({float(pr.get('confidence', 0)):.1f}%)"
                )
            picked_id = st.selectbox(
                "🔎 Open full Doctor & Patient summary for a report",
                options=list(label_map.keys()),
                format_func=lambda i: label_map[i],
                key="history_pick",
            )
            picked = next((r for r in filtered if r.get("id") == picked_id), None)
            if picked is not None:
                render_diagnostic_report(picked, key_prefix="hist")