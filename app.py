import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Union

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

# ============================================================
# 0. STREAMLIT LIGHT THEME
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
                'backgroundColor="#EEF6F9"\n'
                'secondaryBackgroundColor="#FFFFFF"\n'
                'textColor="#0F172A"\n'
                'font="sans serif"\n'
            )
    except Exception:
        pass

load_dotenv()

st.set_page_config(
    page_title="NephroCare AI — Kidney Disease Diagnostic Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. CSS — CLEAN APP-STYLE DESIGN (sidebar nav, stat cards, soft gradients)
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    :root { color-scheme: light !important; }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #EAF6FA 0%, #F4FAF8 55%, #E8F5F1 100%) !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1280px !important; }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1.5px solid #DDEAF2 !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] * { color: #0F172A; }
    .brand {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.4rem 0.4rem 1rem 0.4rem;
        border-bottom: 1.5px solid #EDF3F8; margin-bottom: 0.9rem;
    }
    .brand-logo {
        width: 40px; height: 40px; border-radius: 12px; font-size: 1.3rem;
        background: linear-gradient(135deg, #0284C7, #0D9488);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 8px 16px -6px rgba(2,132,199,0.55);
    }
    .brand b { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em; }
    .brand small { display: block; font-size: 0.68rem; color: #64748B; font-weight: 600; }

    /* Nav radio -> sidebar menu */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        background: transparent !important; border: none !important; padding: 0 !important;
        gap: 0.3rem; display: flex; flex-direction: column;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%; padding: 0.7rem 0.95rem; border-radius: 12px; cursor: pointer;
        border: 1.5px solid transparent; transition: all 0.18s ease; margin: 0;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-weight: 700 !important; font-size: 0.94rem !important; color: #334155 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #F0F7FC; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #E0F2FE, #DDF6F1);
        border-color: #BAE6FD;
        box-shadow: 0 6px 14px -8px rgba(2,132,199,0.5);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: #0369A1 !important; }

    /* ---------- CARDS / CONTAINERS ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1.5px solid #D9E8F2 !important;
        border-radius: 22px !important;
        padding: 1.4rem 1.5rem !important;
        box-shadow: 0 18px 36px -24px rgba(15,23,42,0.22) !important;
    }
    .card {
        background: #FFFFFF; border: 1.5px solid #D9E8F2; border-radius: 22px;
        padding: 1.4rem 1.6rem; box-shadow: 0 18px 36px -24px rgba(15,23,42,0.22);
        animation: fadeUp 0.5s ease both; height: 100%;
    }
    .card-title { font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 0 0 0.25rem 0; }
    .card-sub { font-size: 0.9rem; color: #64748B; margin: 0 0 1rem 0; }
    .pill-tag {
        display: inline-block; background: #E0F2FE; color: #0369A1; font-size: 0.72rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.05em; padding: 0.28rem 0.8rem; border-radius: 999px; margin-bottom: 0.5rem;
    }
    .page-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; color: #0F172A; }
    .page-sub { color: #64748B; font-weight: 500; margin: 0 0 1.2rem 0; }

    /* ---------- INPUTS ---------- */
    div[data-baseweb="textarea"], div[data-baseweb="base-input"], div[data-baseweb="input"] {
        background-color: #FFFFFF !important; border: 2px solid #BDD2E4 !important; border-radius: 14px !important;
    }
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        background-color: #FFFFFF !important; color: #0F172A !important; -webkit-text-fill-color: #0F172A !important;
        caret-color: #0284C7 !important; font-weight: 500 !important; border-radius: 12px !important;
    }
    div[data-baseweb="textarea"]:focus-within, div[data-baseweb="base-input"]:focus-within, div[data-baseweb="input"]:focus-within {
        border-color: #0284C7 !important; box-shadow: 0 0 0 4px rgba(2,132,199,0.15) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important; color: #0F172A !important; border: 2px solid #BDD2E4 !important;
        border-radius: 12px !important; font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #F8FBFF !important; border: 2px dashed #7DD3FC !important; border-radius: 16px !important;
    }
    .stTextArea label p, .stTextInput label p, .stNumberInput label p, .stSelectbox label p, .stFileUploader label p {
        color: #0F172A !important; font-weight: 700 !important; font-size: 0.9rem !important;
    }
    .main div[role="radiogroup"] {
        background: #F0F7FC !important; padding: 0.45rem 1rem !important; border-radius: 14px !important; border: 1.5px solid #CFE0EE !important;
    }
    .main div[role="radiogroup"] label p { color: #0F172A !important; font-weight: 700 !important; font-size: 0.9rem !important; }

    /* ---------- BUTTONS ---------- */
    div.stButton > button[kind="secondary"] {
        background-color: #F0F7FC !important; color: #0369A1 !important; border: 2px solid #BAE6FD !important;
        border-radius: 14px !important; padding: 0.6rem 1rem !important; font-weight: 700 !important; transition: all 0.18s ease !important;
    }
    div.stButton > button[kind="secondary"] * { color: #0369A1 !important; }
    div.stButton > button[kind="secondary"]:hover { background-color: #E0F2FE !important; border-color: #0284C7 !important; transform: translateY(-1px); }
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%) !important; color: #FFFFFF !important; border: none !important;
        border-radius: 14px !important; padding: 0.8rem 1.5rem !important; font-weight: 800 !important;
        box-shadow: 0 10px 22px -8px rgba(2,132,199,0.5) !important; transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"] *, div[data-testid="stFormSubmitButton"] > button * { color: #FFFFFF !important; }
    div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-2px); }
    .st-key-hero_cta button, .st-key-cta_go_patient button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        box-shadow: 0 12px 24px -8px rgba(5,150,105,0.55) !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background-color: #FFFFFF !important; color: #0F172A !important; border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important; font-weight: 700 !important;
    }
    div[data-testid="stDownloadButton"] > button:hover { border-color: #0284C7 !important; background-color: #F0F9FF !important; }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important; border: 2px solid #D8E6F2 !important; border-radius: 14px !important;
        padding: 0.6rem 1.3rem !important; margin-right: 0.6rem !important; font-weight: 700 !important; color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0284C7, #0D9488) !important; color: #FFFFFF !important; border-color: #0284C7 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF !important; }

    /* ---------- ANIMATIONS ---------- */
    @keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes floatGlow { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(22px,-16px) scale(1.1); } }
    @keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    @keyframes ringPop { from { opacity: 0; transform: scale(0.8) rotate(-30deg); } to { opacity: 1; transform: scale(1) rotate(0); } }
    @keyframes orbPulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.07); } }
    @keyframes borderFlow { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
    @keyframes dotBounce { 0%,80%,100% { transform: translateY(0); opacity: 0.35; } 40% { transform: translateY(-6px); opacity: 1; } }
    @keyframes chipPop { from { opacity: 0; transform: scale(0.7) translateY(10px); } to { opacity: 1; transform: scale(1) translateY(0); } }
    @keyframes packetMove { 0% { left: 0; opacity: 0; } 15% { opacity: 1; } 85% { opacity: 1; } 100% { left: calc(100% - 12px); opacity: 0; } }
    @keyframes nodeScan { 0% { top: -8px; opacity: 0; } 15% { opacity: 1; } 85% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
    @keyframes treeBlink { 0%,100% { transform: translateY(0) scale(1); opacity: 0.45; } 50% { transform: translateY(-4px) scale(1.25); opacity: 1; } }
    @keyframes scanDown { 0% { top: -70px; opacity: 0; } 12% { opacity: 1; } 88% { opacity: 1; } 100% { top: 100%; opacity: 0; } }

    /* ---------- HOME HERO ---------- */
    .home-hero {
        position: relative; overflow: hidden; isolation: isolate;
        display: grid; grid-template-columns: 1fr 1.1fr; gap: 1.5rem; align-items: center;
        padding: 2.4rem 2.6rem; border-radius: 32px;
        background: linear-gradient(145deg, #FFFFFF 0%, #EAF6FB 55%, #E3F6F0 100%);
        border: 1.5px solid #FFFFFF;
        box-shadow: 0 40px 70px -32px rgba(7,89,133,0.38), 0 14px 26px -14px rgba(15,23,42,0.12);
        animation: fadeUp 0.6s ease both;
    }
    .home-hero::before, .home-hero::after { content: ""; position: absolute; border-radius: 50%; filter: blur(60px); z-index: -1; }
    .home-hero::before { width: 360px; height: 360px; top: -130px; left: -90px; background: radial-gradient(circle, rgba(56,189,248,0.4), rgba(56,189,248,0) 70%); animation: floatGlow 11s ease-in-out infinite; }
    .home-hero::after { width: 380px; height: 380px; bottom: -150px; right: -100px; background: radial-gradient(circle, rgba(45,212,191,0.38), rgba(45,212,191,0) 70%); animation: floatGlow 14s ease-in-out infinite reverse; }
    .home-art { display: flex; justify-content: center; animation: floaty 5s ease-in-out infinite; }
    .home-art svg { width: 100%; max-width: 280px; height: auto; filter: drop-shadow(0 22px 26px rgba(225,29,72,0.25)); }
    .home-hero h1 { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.025em; margin: 0.3rem 0 0.2rem 0; color: #0F172A; }
    .home-hero h3 { font-size: 1.05rem; font-weight: 700; color: #0F766E; margin: 0.2rem 0 0.8rem 0; }
    .home-hero p { color: #475569; font-weight: 500; line-height: 1.65; margin: 0 0 0.5rem 0; }
    .feat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin-top: 1.2rem; }
    .feat-card {
        background: #FFFFFF; border: 1.5px solid #D9E8F2; border-radius: 20px; padding: 1.2rem 1.3rem;
        box-shadow: 0 16px 30px -22px rgba(15,23,42,0.25); transition: transform 0.25s ease, box-shadow 0.25s ease; animation: fadeUp 0.7s ease both;
    }
    .feat-card:hover { transform: translateY(-5px); box-shadow: 0 26px 40px -22px rgba(2,132,199,0.4); }
    .feat-card .fi { font-size: 1.6rem; width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: #E0F2FE; margin-bottom: 0.6rem; }
    .feat-card b { display: block; font-size: 0.98rem; margin-bottom: 0.2rem; }
    .feat-card span { color: #64748B; font-size: 0.86rem; line-height: 1.5; }

    /* ---------- DASHBOARD STAT CARDS ---------- */
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin-bottom: 1.2rem; }
    .stat-card {
        display: flex; align-items: center; gap: 1rem; padding: 1.2rem 1.4rem; border-radius: 20px;
        background: #FFFFFF; border: 1.5px solid #D9E8F2; box-shadow: 0 16px 30px -22px rgba(15,23,42,0.25);
        transition: transform 0.25s ease, box-shadow 0.25s ease; animation: fadeUp 0.5s ease both;
    }
    .stat-card:hover { transform: translateY(-4px); box-shadow: 0 24px 38px -22px rgba(2,132,199,0.35); }
    .stat-ico { width: 52px; height: 52px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; }
    .stat-ico.blue { background: #E0F2FE; } .stat-ico.green { background: #D1FAE5; } .stat-ico.red { background: #FFE4E6; } .stat-ico.teal { background: #CCFBF1; }
    .stat-lbl { font-size: 0.74rem; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-val { font-size: 1.9rem; font-weight: 800; line-height: 1.1; color: #0F172A; }
    .stat-val.green { color: #059669; } .stat-val.red { color: #E11D48; } .stat-val.blue { color: #0284C7; }

    .pred-box { border-radius: 18px; padding: 1.3rem 1.2rem; text-align: center; border: 2px solid; margin-top: 0.4rem; }
    .pred-box.ok { background: linear-gradient(135deg, #F0FDF9, #DCFCE7); border-color: #6EE7B7; }
    .pred-box.ckd { background: linear-gradient(135deg, #FFF7F8, #FFE4E6); border-color: #FDA4AF; }
    .pred-box h2 { margin: 0.2rem 0; font-size: 2rem; font-weight: 800; }
    .pred-box.ok h2 { color: #047857 !important; } .pred-box.ckd h2 { color: #BE123C !important; }
    .pred-box p { margin: 0; font-weight: 700; color: #475569; }
    .conf-bar { height: 12px; border-radius: 999px; background: #DCEBF6; overflow: hidden; margin-top: 1rem; box-shadow: inset 0 2px 4px rgba(15,23,42,0.1); }
    .conf-bar > div { height: 100%; border-radius: 999px; }
    .conf-bar > div.ok { background: linear-gradient(90deg, #34D399, #059669); }
    .conf-bar > div.ckd { background: linear-gradient(90deg, #FB7185, #E11D48); }

    /* ---------- READABLE TEXT ---------- */
    .readable-paragraph { font-size: 0.97rem; line-height: 1.72; color: #1E293B; margin: 0.4rem 0; }
    .readable-list { margin: 0.4rem 0 0 1.2rem; padding: 0; font-size: 0.95rem; line-height: 1.68; color: #1E293B; }
    .readable-list li { margin-bottom: 0.35rem; }
    .patient-para { font-size: 1.03rem !important; line-height: 1.85 !important; }

    /* ---------- SUMMARY STAGE ---------- */
    .summary-stage {
        position: relative; overflow: hidden; isolation: isolate; margin-top: 1.2rem; padding: 1.6rem; border-radius: 30px;
        background: linear-gradient(145deg, #F8FCFF 0%, #EAF4FB 52%, #E4F6F2 100%);
        border: 1.5px solid rgba(255,255,255,0.95);
        box-shadow: 0 36px 64px -30px rgba(7,89,133,0.34), 0 16px 28px -16px rgba(15,23,42,0.14);
        animation: fadeUp 0.6s cubic-bezier(.2,.8,.2,1) both;
    }
    .summary-stage::before { content: ""; position: absolute; width: 360px; height: 360px; top: -130px; left: -100px; border-radius: 50%; filter: blur(60px); z-index: -1; background: radial-gradient(circle, rgba(56,189,248,0.4), rgba(56,189,248,0) 70%); animation: floatGlow 11s ease-in-out infinite; }
    .sum-hero {
        display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; padding: 1.3rem 1.6rem; border-radius: 24px; position: relative;
        background: rgba(255,255,255,0.8); backdrop-filter: blur(14px); border: 1.5px solid #FFFFFF;
        box-shadow: 0 20px 40px -18px rgba(15,23,42,0.18); margin-bottom: 1.3rem;
    }
    .sum-hero::before { content: ""; position: absolute; left: 0; top: 18px; bottom: 18px; width: 7px; border-radius: 0 8px 8px 0; }
    .sum-hero-ckd::before { background: linear-gradient(180deg, #FB7185, #E11D48); }
    .sum-hero-ok::before { background: linear-gradient(180deg, #34D399, #059669); }
    .conf-ring { width: 124px; height: 124px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; animation: ringPop 0.9s cubic-bezier(.2,.9,.3,1.2) both; box-shadow: 0 16px 30px -10px rgba(15,23,42,0.28); }
    .conf-ring-inner { width: 94px; height: 94px; border-radius: 50%; background: #FFFFFF; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: inset 0 3px 8px rgba(15,23,42,0.08); }
    .conf-ring-inner strong { font-size: 1.4rem; font-weight: 800; line-height: 1.1; }
    .conf-ring-inner span { font-size: 0.6rem; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
    .sum-hero-text { flex: 1; min-width: 260px; }
    .risk-chip { display: inline-block; font-size: 0.74rem; font-weight: 800; padding: 0.3rem 0.9rem; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em; }
    .risk-chip-ckd { background: #FFE4E6; color: #BE123C; } .risk-chip-ok { background: #D1FAE5; color: #047857; }
    .sum-hero-text h2 { margin: 0.55rem 0 0.7rem 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.015em; }
    .sum-hero-ckd h2 { color: #881337 !important; } .sum-hero-ok h2 { color: #064E3B !important; }
    .meta-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .meta-chip { background: #F1F5F9; border: 1.5px solid #E2E8F0; color: #334155; font-size: 0.82rem; font-weight: 700; padding: 0.33rem 0.85rem; border-radius: 999px; }
    .meta-chip-warn { background: #FFF1F2; border-color: #FDA4AF; color: #9F1239; }
    .meta-chip-good { background: #ECFDF5; border-color: #6EE7B7; color: #065F46; }
    .sum-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.3rem; }
    .sum-card { position: relative; display: flex; flex-direction: column; padding: 1.6rem 1.7rem; border-radius: 26px; background: rgba(255,255,255,0.93); overflow: hidden; transition: transform 0.3s ease, box-shadow 0.3s ease; animation: fadeUp 0.7s ease both; }
    .sum-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 6px; }
    .sum-card-doc { box-shadow: 0 28px 48px -26px rgba(2,132,199,0.38); } .sum-card-doc::before { background: linear-gradient(90deg, #0284C7, #38BDF8); }
    .sum-card-pat { box-shadow: 0 28px 48px -26px rgba(13,148,136,0.38); } .sum-card-pat::before { background: linear-gradient(90deg, #0D9488, #2DD4BF); }
    .sum-card:hover { transform: translateY(-6px); }
    .sum-card-head { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.4rem; }
    .sum-icon { width: 56px; height: 56px; flex-shrink: 0; border-radius: 18px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 12px 22px -8px rgba(15,23,42,0.3); }
    .sum-icon-doc { background: linear-gradient(135deg, #38BDF8, #0284C7); } .sum-icon-pat { background: linear-gradient(135deg, #5EEAD4, #0D9488); }
    .sum-section { margin-top: 1.1rem; padding: 1rem 1.2rem; border-radius: 18px; background: linear-gradient(135deg, #F8FAFC, #F1F5F9); border: 1.5px solid #E2E8F0; border-left: 5px solid #0284C7; }
    .sum-section-warn { background: linear-gradient(135deg, #FFF7F8, #FFF1F2); border-color: #FECDD3; border-left-color: #E11D48; }
    .sum-section-teal { background: linear-gradient(135deg, #F7FEFC, #F0FDFA); border-color: #CCFBF1; border-left-color: #0D9488; }
    .sum-section-title { font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
    .sum-card .readable-list { list-style: none; margin: 0.4rem 0 0 0; padding: 0; }
    .sum-card .readable-list li { position: relative; padding-left: 1.85rem; margin-bottom: 0.5rem; }
    .sum-card .readable-list li::before { content: "✓"; position: absolute; left: 0; top: 0.18rem; width: 1.2rem; height: 1.2rem; border-radius: 50%; background: #DCFCE7; color: #166534; font-size: 0.7rem; font-weight: 800; display: flex; align-items: center; justify-content: center; }
    .sum-section-warn .readable-list li::before { content: "!"; background: #FFE4E6; color: #BE123C; }
    .sum-section-teal .readable-list li::before { background: #CCFBF1; color: #0F766E; }

    /* ---------- SNAPSHOT / BIOMARKERS (V2 REFINED CLINICAL TELEMETRY) ---------- */
    .snapshot-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAFDFF 100%);
        border: 1.5px solid #D9E8F2;
        border-radius: 24px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 24px 48px -26px rgba(2, 132, 199, 0.22);
        animation: fadeUp 0.5s ease both;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .patient-banner-v2 {
        background: linear-gradient(135deg, #F0F9FF 0%, #E6F7F4 100%);
        border: 1.5px solid #BAE6FD;
        border-radius: 18px;
        padding: 0.95rem 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.8rem 0 1.1rem 0;
        box-shadow: 0 8px 18px -8px rgba(2, 132, 199, 0.16);
    }
    .pb-left {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .pb-avatar {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0284C7, #0D9488);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        color: #FFFFFF;
        box-shadow: 0 6px 14px -4px rgba(2, 132, 199, 0.45);
    }
    .pb-eyebrow {
        font-size: 0.64rem;
        font-weight: 800;
        color: #0369A1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }
    .pb-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .pb-age-chip {
        background: #E0F2FE;
        color: #0369A1;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
    }
    .pb-stat-badge {
        background: #FFFFFF;
        border: 1.5px solid #BAE6FD;
        border-radius: 14px;
        padding: 0.5rem 0.95rem;
        text-align: right;
        box-shadow: 0 4px 10px -4px rgba(2, 132, 199, 0.12);
    }
    .pb-stat-lbl {
        display: block;
        font-size: 0.62rem;
        font-weight: 800;
        color: #0284C7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.1rem;
    }
    .pb-stat-val b {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0284C7;
    }
    .pb-stat-val span {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
        margin-left: 0.2rem;
    }

    .bio-metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        flex: 1;
    }
    .bio-metric-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 0.9rem 1.05rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 104px;
        transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .bio-metric-card:hover {
        transform: translateY(-2.5px);
        box-shadow: 0 14px 28px -10px rgba(15, 23, 42, 0.12);
    }
    .bio-metric-card.flag {
        border: 1.5px solid #FDA4AF;
        background: linear-gradient(145deg, #FFFFFF 0%, #FFF5F6 100%);
        box-shadow: 0 4px 14px -6px rgba(225, 29, 72, 0.12);
        border-left: 4.5px solid #E11D48;
    }
    .bio-metric-card.ok {
        border: 1.5px solid #E2E8F0;
        background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
        box-shadow: 0 4px 14px -6px rgba(15, 23, 42, 0.05);
        border-left: 4.5px solid #059669;
    }
    .bmc-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .bmc-label-wrap {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        min-width: 0;
        flex: 1;
    }
    .bmc-icon {
        font-size: 0.95rem;
        line-height: 1;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 7px;
        background: #F1F5F9;
        flex-shrink: 0;
    }
    .bio-metric-card.flag .bmc-icon {
        background: #FFE4E6;
    }
    .bio-metric-card.ok .bmc-icon {
        background: #DCFCE7;
    }
    .bmc-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.01em;
        line-height: 1.25;
        word-break: normal;
    }
    .bmc-badge {
        font-size: 0.62rem;
        font-weight: 800;
        padding: 0.16rem 0.52rem;
        border-radius: 999px;
        letter-spacing: 0.02em;
        flex-shrink: 0;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .bmc-badge.flag {
        background: #FFE4E6;
        color: #BE123C;
        border: 1px solid #FECDD3;
    }
    .bmc-badge.ok {
        background: #DCFCE7;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .bmc-middle {
        display: flex;
        align-items: baseline;
        gap: 0.35rem;
        margin: 0.2rem 0 0.35rem 0;
    }
    .bmc-val {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    .bmc-val.flag {
        color: #BE123C;
    }
    .bmc-val.ok {
        color: #0F172A;
    }
    .bmc-unit {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
    }
    .bmc-bottom {
        display: flex;
        align-items: center;
        margin-top: auto;
        padding-top: 0.2rem;
    }
    .bmc-ref-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(241, 245, 249, 0.85);
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.15rem 0.45rem;
        font-size: 0.66rem;
    }
    .bmc-ref-tag .ref-lbl {
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        font-size: 0.58rem;
        letter-spacing: 0.03em;
    }
    .bmc-ref-tag .ref-num {
        font-weight: 700;
        color: #475569;
    }

    .snapshot-footer-bar {
        margin-top: 1rem;
        border-radius: 16px;
        padding: 0.75rem 1.15rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .snapshot-footer-bar.has-abnormal {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1.5px solid #FECDD3;
        color: #9F1239;
        box-shadow: 0 4px 12px -4px rgba(225, 29, 72, 0.12);
    }
    .snapshot-footer-bar.all-normal {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1.5px solid #A7F3D0;
        color: #065F46;
        box-shadow: 0 4px 12px -4px rgba(5, 150, 105, 0.12);
    }
    .sf-status-pill {
        background: #FFFFFF;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.66rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        box-shadow: 0 2px 6px -2px rgba(15, 23, 42, 0.08);
    }
    .snapshot-footer-bar.has-abnormal .sf-status-pill { color: #BE123C; }
    .snapshot-footer-bar.all-normal .sf-status-pill { color: #059669; }

    /* ---------- DATA SCIENCE ANALYTICS SUITE ---------- */
    .ds-kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.9rem;
        margin: 1.1rem 0 1.5rem 0;
    }
    .ds-kpi-card {
        background: #FFFFFF;
        border: 1.5px solid #D9E8F2;
        border-radius: 20px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 24px -12px rgba(2, 132, 199, 0.14);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 104px;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        position: relative;
        overflow: hidden;
    }
    .ds-kpi-card:hover {
        transform: translateY(-2.5px);
        box-shadow: 0 16px 32px -12px rgba(15, 23, 42, 0.16);
    }
    .ds-kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
    }
    .ds-kpi-card.blue::before { background: linear-gradient(90deg, #0284C7, #38BDF8); }
    .ds-kpi-card.red::before { background: linear-gradient(90deg, #E11D48, #FDA4AF); }
    .ds-kpi-card.green::before { background: linear-gradient(90deg, #059669, #34D399); }
    .ds-kpi-card.teal::before { background: linear-gradient(90deg, #0D9488, #2DD4BF); }
    .ds-kpi-card.purple::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
    .ds-kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.35rem;
    }
    .ds-kpi-title {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .ds-kpi-icon {
        font-size: 1.15rem;
        line-height: 1;
    }
    .ds-kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .ds-kpi-val.red { color: #BE123C; }
    .ds-kpi-val.green { color: #065F46; }
    .ds-kpi-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
        line-height: 1.25;
    }
    .ds-findings-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #F0F9FF 100%);
        border: 1.5px solid #BAE6FD;
        border-radius: 20px;
        padding: 1.15rem 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px -8px rgba(2, 132, 199, 0.12);
    }
    .ds-findings-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #0369A1;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .ds-findings-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
    }
    .ds-finding-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.75rem 0.95rem;
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.4;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .ds-finding-icon {
        font-size: 1.1rem;
        line-height: 1.2;
        flex-shrink: 0;
    }


    .bio-panel-box { background: #FFFFFF; border: 1.5px solid #D9E8F2; border-radius: 24px; padding: 1.5rem 1.7rem; margin-top: 1.3rem; box-shadow: 0 22px 40px -26px rgba(15,23,42,0.2); animation: fadeUp 0.7s ease both; }
    .bio-grid-20 { display: grid; grid-template-columns: repeat(auto-fill, minmax(195px, 1fr)); gap: 0.8rem; margin-top: 1rem; }
    .bio-tile-ok, .bio-tile-flag { border-radius: 16px; padding: 0.85rem 1rem; transition: transform 0.25s ease, box-shadow 0.25s ease; }
    .bio-tile-ok { background: #F8FAFC; border: 1.5px solid #E2E8F0; } .bio-tile-flag { background: #FFF1F2; border: 2px solid #FDA4AF; }
    .bio-tile-ok:hover, .bio-tile-flag:hover { transform: translateY(-3px); box-shadow: 0 14px 24px -10px rgba(15,23,42,0.2); }
    .badge-ok { background: #DCFCE7; color: #166534; font-size: 0.68rem; font-weight: 800; padding: 0.2rem 0.55rem; border-radius: 999px; }
    .badge-flag { background: #FFE4E6; color: #9F1239; font-size: 0.68rem; font-weight: 800; padding: 0.2rem 0.55rem; border-radius: 999px; }

    /* ---------- PIPELINE FLOW PANEL (popup) ---------- */
    .ai-think { position: relative; overflow: hidden; isolation: isolate; display: block; margin-top: 0.8rem; padding: 1.4rem 1.5rem 1.2rem; border-radius: 26px;
        background: linear-gradient(135deg, #F8FCFF 0%, #EAF4FB 55%, #E4F6F2 100%); border: 2px solid #BAE6FD;
        box-shadow: 0 30px 56px -28px rgba(2,132,199,0.45); animation: fadeUp 0.45s cubic-bezier(.2,.8,.2,1) both; }
    .ai-think::before { content: ""; position: absolute; inset: 0; z-index: -1; background: linear-gradient(110deg, rgba(56,189,248,0.22), rgba(45,212,191,0.22), rgba(56,189,248,0.22)); background-size: 200% 100%; animation: shimmer 6s linear infinite; }
    .ai-think::after { content: ""; position: absolute; left: 0; right: 0; height: 70px; pointer-events: none; display: none; }
    .ai-think-done { border-color: #6EE7B7; }
    .flow-head { display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.9rem; }
    .flow-head .ai-body { flex: 1; min-width: 0; }
    .ai-core { width: 62px; height: 62px; flex-shrink: 0; border-radius: 50%; background: linear-gradient(135deg, #0284C7, #0D9488); display: flex; align-items: center; justify-content: center; font-size: 1.7rem; box-shadow: 0 16px 28px -10px rgba(2,132,199,0.65); animation: orbPulse 2.2s ease-in-out infinite; }
    .ai-core-done { background: linear-gradient(135deg, #10B981, #059669); animation: none; }
    .ai-title { margin: 0.1rem 0 0.2rem 0; font-size: 1.25rem; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .ai-dots { display: inline-flex; gap: 4px; }
    .ai-dots b { width: 8px; height: 8px; border-radius: 50%; background: #0284C7; display: inline-block; animation: dotBounce 1.2s ease-in-out infinite; }
    .ai-dots b:nth-child(2) { animation-delay: 0.15s; background: #0891B2; } .ai-dots b:nth-child(3) { animation-delay: 0.3s; background: #0D9488; }
    .ai-detail { margin: 0; font-size: 0.92rem; color: #475569; font-weight: 500; }
    .ai-pct { font-size: 1.35rem; font-weight: 800; color: #0369A1; min-width: 64px; text-align: right; }
    .ai-bar-row { display: flex; align-items: center; gap: 0.8rem; }
    .ai-bar { flex: 1; height: 12px; border-radius: 999px; background: #DCEBF6; overflow: hidden; box-shadow: inset 0 2px 4px rgba(15,23,42,0.1); }
    .ai-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0284C7, #2DD4BF, #0284C7); background-size: 200% 100%; animation: borderFlow 1.6s linear infinite; }
    .ai-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.9rem; }
    .ai-chip { background: #FFFFFF; border: 1.5px solid #BAE6FD; color: #0369A1; font-size: 0.8rem; font-weight: 800; padding: 0.3rem 0.8rem; border-radius: 999px; animation: chipPop 0.5s cubic-bezier(.2,.9,.3,1.3) both; }
    .ai-chip-flag { background: #FFF1F2; border-color: #FDA4AF; color: #BE123C; }
    .fl-row { display: flex; align-items: center; gap: 0.15rem; margin-top: 1.1rem; padding: 1rem 0.7rem; border-radius: 22px; background: rgba(255,255,255,0.78); border: 1.5px solid #FFFFFF; overflow-x: auto; }
    .fl-node { position: relative; overflow: hidden; flex: 0 0 92px; text-align: center; padding: 0.7rem 0.3rem 0.6rem; border-radius: 18px; background: #FFFFFF; border: 2px solid #E2E8F0; transition: all 0.3s ease; }
    .fl-ico { position: relative; width: 46px; height: 46px; margin: 0 auto 0.35rem; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; background: #F1F5F9; filter: grayscale(0.6); opacity: 0.7; }
    .fl-name { font-size: 0.74rem; font-weight: 800; color: #0F172A; } .fl-sub { font-size: 0.64rem; font-weight: 600; color: #64748B; min-height: 1.1rem; }
    .fl-node.active { border-color: #38BDF8; transform: translateY(-4px); box-shadow: 0 0 0 4px rgba(56,189,248,0.22), 0 18px 30px -12px rgba(2,132,199,0.6); }
    .fl-node.active .fl-ico { background: linear-gradient(135deg, #0284C7, #0D9488); filter: none; opacity: 1; animation: orbPulse 1.1s ease-in-out infinite; }
    .fl-node.active .fl-sub { color: #0369A1; }
    .fl-node.scanning::after { content: ""; position: absolute; left: 0; right: 0; height: 8px; background: linear-gradient(180deg, rgba(45,212,191,0), rgba(45,212,191,0.9)); animation: nodeScan 1.1s ease-in-out infinite; }
    .fl-node.done { border-color: #6EE7B7; background: #F0FDF9; }
    .fl-node.done .fl-ico { background: #D1FAE5; filter: none; opacity: 1; }
    .fl-node.done .fl-ico::after { content: "✓"; position: absolute; right: -7px; bottom: -7px; width: 18px; height: 18px; border-radius: 50%; background: #059669; color: #FFFFFF; font-size: 0.65rem; font-weight: 800; display: flex; align-items: center; justify-content: center; }
    .fl-trees { display: inline-flex; gap: 1px; font-size: 0.7rem; } .fl-trees b { font-weight: 400; animation: treeBlink 0.8s ease-in-out infinite; }
    .fl-trees b:nth-child(2) { animation-delay: 0.12s; } .fl-trees b:nth-child(3) { animation-delay: 0.24s; } .fl-trees b:nth-child(4) { animation-delay: 0.36s; } .fl-trees b:nth-child(5) { animation-delay: 0.48s; }
    .fl-conn { position: relative; flex: 1 1 26px; min-width: 26px; height: 12px; }
    .fl-conn::before { content: ""; position: absolute; left: 0; right: 0; top: 4px; height: 4px; border-radius: 4px; background: #DCEBF6; }
    .fl-conn.done::before { background: linear-gradient(90deg, #34D399, #10B981); }
    .fl-conn.sending::before { background: linear-gradient(90deg, #BAE6FD, #38BDF8, #BAE6FD); background-size: 200% 100%; animation: borderFlow 0.7s linear infinite; }
    .fl-conn i { position: absolute; top: 0; left: 0; width: 12px; height: 12px; border-radius: 50%; background: radial-gradient(circle, #5EEAD4, #0284C7); box-shadow: 0 0 10px rgba(2,132,199,0.85); opacity: 0; }
    .fl-conn.sending i { animation: packetMove 0.9s linear infinite; } .fl-conn.sending i:nth-child(2) { animation-delay: 0.3s; } .fl-conn.sending i:nth-child(3) { animation-delay: 0.6s; }
    div[data-testid="stDialog"] div[role="dialog"] { border-radius: 28px !important; background: #F4F9FD !important; border: 2px solid #BAE6FD !important; box-shadow: 0 50px 110px -24px rgba(7,89,133,0.55) !important; }
    div[data-testid="stDialog"] h2 { font-weight: 800 !important; color: #0F172A !important; }
    .cta-note { margin: 1rem 0 0.7rem 0; padding: 1rem 1.3rem; border-radius: 18px; background: linear-gradient(135deg, #ECFDF5, #F0FDFA); border: 2px solid #6EE7B7; color: #065F46; font-weight: 700; text-align: center; }

    
    /* ---------- RURAL HEALTH CAMP TRIAGE BOARD ---------- */
    .triage-hero-box {
        background: linear-gradient(135deg, #075985 0%, #0284C7 50%, #0D9488 100%);
        border-radius: 24px;
        padding: 1.6rem 2rem;
        color: #FFFFFF;
        box-shadow: 0 24px 48px -18px rgba(2, 132, 199, 0.45);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .triage-hero-box::after {
        content: "⛺";
        position: absolute;
        right: 1.5rem;
        bottom: -0.5rem;
        font-size: 7.5rem;
        opacity: 0.12;
        pointer-events: none;
    }
    .triage-hero-box h2 {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0.2rem 0 0.5rem 0;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .triage-hero-box p {
        font-size: 0.92rem;
        color: #E0F2FE;
        max-width: 840px;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .triage-hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
    }
    .triage-meta-pill {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        color: #FFFFFF;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .triage-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .triage-kpi-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 10px 24px -10px rgba(15, 23, 42, 0.12);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1.5px solid #E2E8F0;
        position: relative;
        overflow: hidden;
    }
    .triage-kpi-card:hover {
        transform: translateY(-2.5px);
        box-shadow: 0 16px 30px -10px rgba(15, 23, 42, 0.16);
    }
    .triage-kpi-card.urgent {
        border-color: #FDA4AF;
        background: linear-gradient(145deg, #FFFFFF 0%, #FFF5F6 100%);
        border-left: 5px solid #E11D48;
    }
    .triage-kpi-card.monitor {
        border-color: #FDE68A;
        background: linear-gradient(145deg, #FFFFFF 0%, #FFFDF5 100%);
        border-left: 5px solid #D97706;
    }
    .triage-kpi-card.routine {
        border-color: #A7F3D0;
        background: linear-gradient(145deg, #FFFFFF 0%, #F5FDF9 100%);
        border-left: 5px solid #059669;
    }
    .triage-kpi-card.total {
        border-color: #BAE6FD;
        background: linear-gradient(145deg, #FFFFFF 0%, #F0F9FF 100%);
        border-left: 5px solid #0284C7;
    }

    .tkpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }
    .tkpi-title {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .tkpi-val {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .tkpi-val.urgent { color: #BE123C; }
    .tkpi-val.monitor { color: #D97706; }
    .tkpi-val.routine { color: #059669; }
    .tkpi-val.total { color: #0284C7; }
    .tkpi-sub {
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748B;
    }

    .triage-report-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 14px 30px -14px rgba(15, 23, 42, 0.14);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1.5px solid #E2E8F0;
        transition: transform 0.24s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.24s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 1.2rem;
        min-height: 380px;
    }
    .triage-report-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px -16px rgba(15, 23, 42, 0.2);
    }
    .triage-report-card.urgent {
        border-color: #FDA4AF;
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF9F9 100%);
        border-top: 5.5px solid #E11D48;
    }
    .triage-report-card.monitor {
        border-color: #FDE68A;
        background: linear-gradient(180deg, #FFFFFF 0%, #FFFDF7 100%);
        border-top: 5.5px solid #D97706;
    }
    .triage-report-card.routine {
        border-color: #A7F3D0;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FDFB 100%);
        border-top: 5.5px solid #059669;
    }

    .trc-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.85rem;
    }
    .trc-patient-info {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .trc-avatar {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        flex-shrink: 0;
    }
    .trc-avatar.urgent { background: #FFE4E6; }
    .trc-avatar.monitor { background: #FEF3C7; }
    .trc-avatar.routine { background: #DCFCE7; }
    .trc-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .trc-meta-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.2rem;
    }
    .trc-age-badge {
        background: #F1F5F9;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.12rem 0.5rem;
        border-radius: 6px;
    }
    .trc-id-badge {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94A3B8;
    }

    .trc-tier-badge {
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        white-space: nowrap;
    }
    .trc-tier-badge.urgent {
        background: #FFE4E6;
        color: #9F1239;
        border: 1.5px solid #FDA4AF;
    }
    .trc-tier-badge.monitor {
        background: #FEF3C7;
        color: #92400E;
        border: 1.5px solid #FDE68A;
    }
    .trc-tier-badge.routine {
        background: #DCFCE7;
        color: #065F46;
        border: 1.5px solid #A7F3D0;
    }

    .trc-conf-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #F8FAFC;
        border-radius: 10px;
        padding: 0.45rem 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.76rem;
    }
    .trc-conf-strip span.pred-lbl {
        font-weight: 800;
    }
    .trc-conf-strip.urgent span.pred-lbl { color: #BE123C; }
    .trc-conf-strip.monitor span.pred-lbl { color: #D97706; }
    .trc-conf-strip.routine span.pred-lbl { color: #059669; }
    .trc-conf-strip span.conf-val {
        font-weight: 700;
        color: #475569;
    }

    .trc-bio-matrix {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
        margin: 0.75rem 0;
    }
    .trc-bio-pill {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.45rem 0.6rem;
        display: flex;
        flex-direction: column;
    }
    .trc-bio-pill.abnormal {
        background: #FFF1F2;
        border-color: #FECDD3;
    }
    .trc-bio-lbl {
        font-size: 0.62rem;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .trc-bio-pill.abnormal .trc-bio-lbl {
        color: #BE123C;
    }
    .trc-bio-val {
        font-size: 0.84rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.1rem;
    }
    .trc-bio-pill.abnormal .trc-bio-val {
        color: #BE123C;
    }

    .trc-directive-box {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        font-size: 0.8rem;
        line-height: 1.45;
        margin-top: 0.6rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .trc-directive-box.urgent {
        background: #FFF1F2;
        border: 1.5px solid #FECDD3;
        border-left: 4.5px solid #E11D48;
        color: #881337;
    }
    .trc-directive-box.monitor {
        background: #FFFBEB;
        border: 1.5px solid #FDE68A;
        border-left: 4.5px solid #D97706;
        color: #78350F;
    }
    .trc-directive-box.routine {
        background: #ECFDF5;
        border: 1.5px solid #A7F3D0;
        border-left: 4.5px solid #059669;
        color: #064E3B;
    }
    .trc-directive-icon {
        font-size: 1.15rem;
        line-height: 1.2;
        flex-shrink: 0;
    }
    .trc-directive-text {
        flex: 1;
    }
    .trc-directive-text b {
        font-weight: 800;
        display: block;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }

    .triage-action-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 50%, #ECFDF5 100%);
        border: 2px solid #7DD3FC;
        border-radius: 24px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.2rem;
        box-shadow: 0 16px 36px -18px rgba(2, 132, 199, 0.28);
    }
    .tab-content {
        flex: 1;
        min-width: 300px;
    }
    .tab-tag {
        display: inline-block;
        background: #0284C7;
        color: #FFFFFF;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .tab-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0C4A6E;
        margin: 0 0 0.35rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .tab-desc {
        font-size: 0.86rem;
        color: #334155;
        margin: 0;
        line-height: 1.45;
    }

    /* ---------- REGISTRY TABLE ---------- */
    .reg-wrap { background: #FFFFFF; border: 1.5px solid #D9E8F2; border-radius: 24px; overflow: hidden; margin-top: 0.4rem; box-shadow: 0 26px 46px -28px rgba(2,132,199,0.32); animation: fadeUp 0.6s ease both; }
    .reg-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.6rem; padding: 1.2rem 1.5rem 0.9rem; background: linear-gradient(135deg, #F8FCFF, #EEF7FB); border-bottom: 1.5px solid #E2E8F0; }
    .reg-head .card-title { margin: 0; }
    .reg-count { background: #E0F2FE; color: #0369A1; font-size: 0.82rem; font-weight: 800; padding: 0.4rem 1rem; border-radius: 999px; }
    .reg-scroll { overflow: auto; max-height: 560px; }
    table.reg-table { border-collapse: separate; border-spacing: 0; width: max-content; min-width: 100%; font-size: 0.86rem; color: #1E293B; }
    .reg-table th, .reg-table td { padding: 0.7rem 0.95rem; white-space: nowrap; text-align: left; border-bottom: 1px solid #E8EFF6; }
    .reg-table thead th { position: sticky; z-index: 3; background: #F0F7FC; color: #0F172A; font-weight: 800; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 2px solid #CFE0EE; }
    .reg-table thead small { display: block; font-size: 0.66rem; font-weight: 600; color: #64748B; text-transform: none; letter-spacing: 0; }
    .reg-table thead tr.grp th { top: 0; height: 40px; text-align: center; font-size: 0.78rem; }
    .reg-table thead tr.cols th { top: 41px; }
    th.g-report { background: #E0F2FE !important; color: #075985; } th.g-pred { background: #FFE4E6 !important; color: #9F1239; } th.g-data { background: #CCFBF1 !important; color: #0F766E; }
    .reg-table tbody td { background: #FFFFFF; } .reg-table tbody tr:nth-child(even) td { background: #F8FBFE; } .reg-table tbody tr:hover td { background: #E6F4FD; }
    .reg-table td.cell-flag { background: #FFF1F2 !important; color: #BE123C; font-weight: 800; }
    .reg-table .sticky-1, .reg-table .sticky-2 { position: sticky; z-index: 2; } .reg-table thead .sticky-1, .reg-table thead .sticky-2 { z-index: 5; }
    .reg-table .sticky-1 { left: 0; min-width: 70px; max-width: 70px; } .reg-table .sticky-2 { left: 70px; min-width: 170px; box-shadow: 8px 0 10px -8px rgba(15,23,42,0.18); }
    .reg-table .id-cell { font-weight: 800; color: #0369A1; } .reg-table .name-cell { font-weight: 800; } .reg-table .date-cell { color: #64748B; font-weight: 600; }
    .tbl-badge { display: inline-block; font-size: 0.74rem; font-weight: 800; padding: 0.25rem 0.75rem; border-radius: 999px; }
    .tbl-badge-ckd { background: #FFE4E6; color: #BE123C; border: 1.5px solid #FDA4AF; } .tbl-badge-ok { background: #D1FAE5; color: #047857; border: 1.5px solid #6EE7B7; }
    .conf-mini { display: flex; align-items: center; gap: 0.55rem; min-width: 130px; }
    .conf-mini-track { flex: 1; height: 8px; border-radius: 999px; background: #E2E8F0; overflow: hidden; } .conf-mini-fill { height: 100%; border-radius: 999px; }
    .conf-mini-fill.ckd { background: linear-gradient(90deg, #FB7185, #E11D48); } .conf-mini-fill.ok { background: linear-gradient(90deg, #34D399, #059669); }
    .conf-mini span { font-weight: 800; font-size: 0.82rem; }

    @media (max-width: 800px) { .home-hero { grid-template-columns: 1fr; padding: 1.5rem; } .sum-grid { grid-template-columns: 1fr; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. DATA SCHEMA & CONSTANTS
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
DEFAULT_SCALER_PATH = r"D:\RAG\InnovateX\scaler.pkl"
FALLBACK_SCALER_PATH = "scaler.pkl"
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
    "appet": {"poor": 0, "good": 1},
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

CKD_DEMO = (
    "Patient Name: Ahmed Raza, Age 62. Presents with pedal edema and poor appetite. "
    "History of hypertension (htn: yes) and diabetes mellitus (dm: yes), no coronary artery disease. "
    "Lab findings: BP 100 mm/Hg, Albumin 3, Sugar 2, RBC abnormal, Pus cells abnormal, "
    "pus cell clumps present, bacteria notpresent. Random blood glucose 240 mg/dL, "
    "Blood Urea 86 mg/dL, Serum Creatinine 3.8 mg/dL, Hemoglobin 9.4 g/dL, PCV 29%, "
    "RBC count 3.4 million/cmm. Patient also has anemia."
)
HEALTHY_DEMO = (
    "Patient Name: Sara Khan, 34 years old. Routine wellness checkup. Blood pressure 76 mm/Hg, "
    "albumin 0, sugar 0, red blood cells normal, pus cells normal, no clumps or bacteria. "
    "Random blood glucose 98 mg/dL, blood urea 24 mg/dL, serum creatinine 0.8 mg/dL, "
    "hemoglobin 14.6 g/dL, PCV 44%, RBC count 5.1 million/cmm. No hypertension, "
    "no diabetes, no coronary artery disease, good appetite, no pedal edema or anemia."
)



RURAL_CAMP_10_DATA = [
    {
        "name": "Tariq Mehmood", "age": 65, "bp": 160, "sc": 4.8, "bu": 112, "hemo": 7.8, "al": 3, "su": 2,
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent", "bgr": 250, "sod": 130,
        "pot": 5.4, "pcv": 24, "wc": 11000, "rc": 3.1, "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor",
        "pe": "yes", "ane": "yes"
    },
    {
        "name": "Fatima Bibi", "age": 58, "bp": 145, "sc": 3.2, "bu": 78, "hemo": 9.1, "al": 2, "su": 1,
        "rbc": "abnormal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 210, "sod": 134,
        "pot": 4.9, "pcv": 28, "wc": 8900, "rc": 3.5, "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor",
        "pe": "yes", "ane": "yes"
    },
    {
        "name": "Muhammad Aslam", "age": 72, "bp": 155, "sc": 5.1, "bu": 130, "hemo": 8.4, "al": 3, "su": 2,
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present", "bgr": 270, "sod": 128,
        "pot": 5.6, "pcv": 26, "wc": 12500, "rc": 2.9, "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor",
        "pe": "yes", "ane": "yes"
    },
    {
        "name": "Rasheeda Begum", "age": 60, "bp": 140, "sc": 2.8, "bu": 68, "hemo": 9.5, "al": 2, "su": 0,
        "rbc": "abnormal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 160, "sod": 135,
        "pot": 4.6, "pcv": 30, "wc": 7500, "rc": 3.7, "htn": "yes", "dm": "no", "cad": "no", "appet": "poor",
        "pe": "yes", "ane": "yes"
    },
    {
        "name": "Abdul Ghafoor", "age": 54, "bp": 135, "sc": 1.8, "bu": 48, "hemo": 11.2, "al": 1, "su": 1,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 185, "sod": 137,
        "pot": 4.4, "pcv": 34, "wc": 7200, "rc": 4.0, "htn": "yes", "dm": "yes", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    },
    {
        "name": "Naseem Akhtar", "age": 49, "bp": 130, "sc": 1.5, "bu": 42, "hemo": 11.8, "al": 1, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 145, "sod": 138,
        "pot": 4.3, "pcv": 36, "wc": 6800, "rc": 4.2, "htn": "no", "dm": "yes", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    },
    {
        "name": "Ghulam Abbas", "age": 52, "bp": 138, "sc": 1.6, "bu": 45, "hemo": 12.0, "al": 1, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 130, "sod": 139,
        "pot": 4.2, "pcv": 37, "wc": 6500, "rc": 4.3, "htn": "yes", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    },
    {
        "name": "Zahid Khan", "age": 32, "bp": 78, "sc": 0.9, "bu": 24, "hemo": 15.2, "al": 0, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 92, "sod": 141,
        "pot": 4.1, "pcv": 45, "wc": 6200, "rc": 5.2, "htn": "no", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    },
    {
        "name": "Amina Sheikh", "age": 28, "bp": 74, "sc": 0.7, "bu": 20, "hemo": 13.8, "al": 0, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 88, "sod": 140,
        "pot": 4.0, "pcv": 42, "wc": 5800, "rc": 4.8, "htn": "no", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    },
    {
        "name": "Bilal Ahmed", "age": 36, "bp": 80, "sc": 1.0, "bu": 26, "hemo": 14.9, "al": 0, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 96, "sod": 142,
        "pot": 4.2, "pcv": 44, "wc": 6400, "rc": 5.1, "htn": "no", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    },
]

RURAL_CAMP_10_TEXT = "\n\n".join([
    f"Patient {i+1}: {p['name']}, Age {p['age']}. BP {p['bp']} mm/Hg, Serum Creatinine {p['sc']} mg/dL, "
    f"Blood Urea {p['bu']} mg/dL, Hemoglobin {p['hemo']} g/dL, Urine Albumin {p['al']}, Urine Sugar {p['su']}, "
    f"RBC {p['rbc']}, Pus Cells {p['pc']}, Pus cell clumps {p['pcc']}, Bacteria {p['ba']}, Blood Glucose {p['bgr']} mg/dL, "
    f"Sodium {p['sod']} mEq/L, Potassium {p['pot']} mEq/L, PCV {p['pcv']}%, RBC count {p['rc']} M/cmm, "
    f"Hypertension {p['htn']}, Diabetes {p['dm']}, CAD {p['cad']}, Appetite {p['appet']}, Pedal Edema {p['pe']}, Anemia {p['ane']}."
    for i, p in enumerate(RURAL_CAMP_10_DATA)
])


def assign_triage_tier(entry: dict) -> dict:
    ud = entry.get("user_data", {})
    pred = entry.get("prediction", {})
    is_ckd = pred.get("value") == 1
    conf = float(pred.get("confidence", 0))

    def _num(val):
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    sc = _num(ud.get("sc"))
    bu = _num(ud.get("bu"))
    hemo = _num(ud.get("hemo"))
    bp = _num(ud.get("bp"))
    al = _num(ud.get("al")) or 0

    if is_ckd and ((sc is not None and sc >= 2.5) or (bu is not None and bu >= 70) or al >= 2 or (hemo is not None and hemo < 10.0) or conf >= 85.0):
        return {
            "tier_key": "urgent",
            "tier_rank": 1,
            "tier_lbl": "Urgent Referral",
            "badge_icon": "🔴",
            "color": "#BE123C",
            "border_color": "#E11D48",
            "bg_color": "#FFF1F2",
            "badge_cls": "urgent",
            "directive_title": "TERTIARY NEPHROLOGY REFERRAL",
            "directive": "Immediate tertiary hospital dispatch. Severe azotemia with metabolic impairment. Avoid NSAIDs / nephrotoxins; initiate fluid & electrolyte stabilization."
        }
    elif is_ckd or (sc is not None and sc >= 1.4) or (bu is not None and bu >= 42) or al >= 1 or (bp is not None and bp >= 90):
        return {
            "tier_key": "monitor",
            "tier_rank": 2,
            "tier_lbl": "Priority Monitor",
            "badge_icon": "🟡",
            "color": "#D97706",
            "border_color": "#D97706",
            "bg_color": "#FFFBEB",
            "badge_cls": "monitor",
            "directive_title": "PRIMARY CLINIC OBSERVATION",
            "directive": "Enroll in Primary Health Center (PHC) chronic monitoring. Optimize BP control (< 130/80 mmHg) and glycemic stabilization. Repeat renal panel in 30 days."
        }
    else:
        return {
            "tier_key": "routine",
            "tier_rank": 3,
            "tier_lbl": "Routine Wellness",
            "badge_icon": "🟢",
            "color": "#059669",
            "border_color": "#059669",
            "bg_color": "#ECFDF5",
            "badge_cls": "routine",
            "directive_title": "ROUTINE CARE CLEARED",
            "directive": "Normal baseline renal filtration. No proteinuria or hematuria detected. Cleared for routine annual community health screening."
        }


def load_rural_camp_demo_cohort(ml_model) -> list:
    cohort_entries = []
    for idx, p_raw in enumerate(RURAL_CAMP_10_DATA):
        clean = sanitize_user_data(p_raw)
        pred, conf = predict(clean, ml_model)
        tier_info = assign_triage_tier({"user_data": clean, "prediction": {"value": pred, "confidence": conf}})
        
        pname = clean.get("name") or f"Patient #{idx+1}"
        page = clean.get("age", "—")
        if tier_info["tier_key"] == "urgent":
            doc = (
                f"**Doctor:** Urgent tertiary nephrology consultation indicated for {pname} (Age {page}). "
                f"Biochemical profile shows severe retention of nitrogenous wastes (Serum Creatinine {clean.get('sc')} mg/dL, "
                f"Blood Urea {clean.get('bu')} mg/dL) with clinically significant anemia (Hemoglobin {clean.get('hemo')} g/dL) "
                f"and proteinuria (Albumin Grade {clean.get('al')}). "
                f"Field directive: {tier_info['directive']}"
            )
            pat = (
                f"**Patient:** {pname}, your kidney test results show that your kidneys are under severe strain and need immediate "
                f"care by a kidney specialist doctor at the district hospital. Please follow the transfer instructions and do not take pain medications like ibuprofen."
            )
        elif tier_info["tier_key"] == "monitor":
            doc = (
                f"**Doctor:** Early-stage renal observation indicated for {pname} (Age {page}). "
                f"Laboratory findings demonstrate moderate biomarker alterations (Serum Creatinine {clean.get('sc')} mg/dL, "
                f"Blood Urea {clean.get('bu')} mg/dL, Urine Albumin Grade {clean.get('al')}). "
                f"Field directive: {tier_info['directive']}"
            )
            pat = (
                f"**Patient:** {pname}, your kidneys are showing mild early signs of stress. "
                f"You should visit the local clinic to check your blood pressure and blood sugar regularly, drink clean water, and repeat your test in one month."
            )
        else:
            doc = (
                f"**Doctor:** Routine screening for {pname} (Age {page}) reveals completely normal renal function indicators "
                f"(Serum Creatinine {clean.get('sc')} mg/dL, Blood Urea {clean.get('bu')} mg/dL, Hemoglobin {clean.get('hemo')} g/dL). "
                f"No proteinuria or glycosuria identified. Field directive: {tier_info['directive']}"
            )
            pat = (
                f"**Patient:** {pname}, good news! Your kidney screening results are healthy and within normal range. "
                f"Keep up healthy lifestyle habits, drink sufficient water, and come back for routine annual community health screening."
            )

        summary_text = f"{doc}\n\n{pat}"
        entry = save_record(clean, pred, conf, summary_text)
        cohort_entries.append(entry)
    return cohort_entries


# ============================================================
# 3. CACHED MODEL & GROQ LLM CHAINS
# ============================================================
class ModelBundle:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return getattr(self.model, "classes_", [0, 1])


@st.cache_resource(show_spinner=False)
def load_ml_model(model_path: str = DEFAULT_MODEL_PATH, scaler_path: Optional[str] = None):
    # Resolve model file
    m_path = None
    if os.path.exists(model_path):
        m_path = model_path
    elif os.path.exists(FALLBACK_MODEL_PATH):
        m_path = FALLBACK_MODEL_PATH

    if not m_path:
        return None, f"Model file not found at '{model_path}'."

    # Resolve scaler file
    s_path = None
    if scaler_path and os.path.exists(scaler_path):
        s_path = scaler_path
    else:
        m_dir = os.path.dirname(m_path)
        adj_scaler = os.path.join(m_dir, "scaler.pkl") if m_dir else "scaler.pkl"
        if os.path.exists(adj_scaler):
            s_path = adj_scaler
        elif os.path.exists(DEFAULT_SCALER_PATH):
            s_path = DEFAULT_SCALER_PATH
        elif os.path.exists(FALLBACK_SCALER_PATH):
            s_path = FALLBACK_SCALER_PATH

    try:
        model = joblib.load(m_path)
        scaler = joblib.load(s_path) if s_path else None
        return ModelBundle(model, scaler), None
    except Exception as e:
        return None, f"Error loading model/scaler: {e}"


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
# 4. CORE LOGIC & HTML HELPERS
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
    if hasattr(ml_model, "model"):
        model = ml_model.model
        scaler = getattr(ml_model, "scaler", None)
    elif isinstance(ml_model, (list, tuple)) and len(ml_model) >= 2:
        model, scaler = ml_model[0], ml_model[1]
    else:
        model = ml_model
        scaler = getattr(ml_model, "scaler", None)

    data = user_data.copy()
    for k, v in DEFAULTS.items():
        if data.get(k) is None:
            data[k] = v
    for col, mapping in CATEGORICAL_MAP.items():
        val = data.get(col)
        if isinstance(val, (int, float)):
            data[col] = int(val)
        else:
            val_str = str(val).strip().lower()
            data[col] = mapping.get(val_str, 0)

    X_df = pd.DataFrame([data])[FEATURES]
    if scaler is not None:
        try:
            X = scaler.transform(X_df)
        except Exception:
            X = X_df.values
    else:
        X = X_df.values

    raw_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    ckd_idx = classes.index(1) if 1 in classes else 1
    ckd_prob = float(proba[ckd_idx])

    pred = int(raw_pred)
    confidence = float(proba.max() * 100)

    # Clinical safety guard: If user provided elevated renal failure markers,
    # don't allow default healthy markers (hemo, pcv, al) to suppress a true CKD diagnosis
    try:
        user_sc = float(user_data.get("sc")) if user_data.get("sc") is not None else None
    except (ValueError, TypeError):
        user_sc = None
    try:
        user_bu = float(user_data.get("bu")) if user_data.get("bu") is not None else None
    except (ValueError, TypeError):
        user_bu = None
    try:
        user_al = float(user_data.get("al")) if user_data.get("al") is not None else None
    except (ValueError, TypeError):
        user_al = None

    is_clinically_ckd = (
        (user_sc is not None and user_sc >= 2.0)
        or (user_sc is not None and user_sc >= 1.5 and user_bu is not None and user_bu >= 50)
        or (user_al is not None and user_al >= 2)
        or (user_bu is not None and user_bu >= 75)
    )
    if is_clinically_ckd and pred == 0:
        pred = 1
        confidence = max(float(ckd_prob * 100), 80.0)

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
            else:
                out[k] = "yes" if s in ("yes", "y", "true", "present", "positive") else "no"
        else:
            try:
                num = float(re.sub(r"[^0-9.\-]", "", str(v)))
                out[k] = int(num) if num.is_integer() else num
            except ValueError:
                continue
    return Extractor(**out).model_dump()


def parse_patients(raw: str) -> list:
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
    extract_chain, _, _ = get_llm_chains(api_key, model_name)
    result = extract_chain.invoke({"paragraph": text, "format_instructions": EXTRACT_FORMAT})
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
    sections = {"abnormal": "", "drivers": "", "doctor": "", "patient": "", "actions": ""}
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
    if not text:
        return "<p class='readable-paragraph'>No details available.</p>"
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    parts, in_list = [], False
    for line in lines:
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        if esc.startswith(("- ", "* ", "• ")):
            if not in_list:
                parts.append("<ul class='readable-list'>")
                in_list = True
            parts.append(f"<li>{esc[2:].strip()}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<p class='readable-paragraph'>{esc}</p>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


def flat_html(markup: str) -> str:
    return re.sub(r"\s*\n\s*", " ", markup).strip()


def to_single_paragraph(text: str, max_sentences: int = 5) -> str:
    if not text:
        return "No details available."
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    t = re.sub(r"^\s*[-*•]\s*", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    sentences = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(sentences[:max_sentences]).strip()


def resolve_json_path() -> str:
    candidates = [
        JSON_PATH,
        os.path.join(os.path.dirname(__file__), "patient_reports.json"),
        os.path.join(os.path.dirname(__file__), "..", "patient_reports.json"),
        r"D:\RAG\patient_reports.json",
        r"D:\RAG\InnovateX\patient_reports.json",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return JSON_PATH


def load_all_records() -> list:
    path = resolve_json_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
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
# 5. SESSION STATE + NAVIGATION
# ============================================================
NAV_HOME = "🏠  Home"
NAV_DASH = "📊  Dashboard"
NAV_SCREEN = "🩺  New Screening"
NAV_REPORTS = "📁  Reports"
NAV_ANALYSIS = "📈  Analysis"
NAV_ABOUT = "ℹ️  About"
NAV_ITEMS = [NAV_HOME, NAV_DASH, NAV_SCREEN, NAV_REPORTS, NAV_ANALYSIS, NAV_ABOUT]

if "patient_form_data" not in st.session_state:
    st.session_state.patient_form_data = {"name": "", **DEFAULTS}
if "latest_entry" not in st.session_state:
    st.session_state.latest_entry = None
if "extracted_raw" not in st.session_state:
    st.session_state.extracted_raw = None
if "clinical_note_input" not in st.session_state:
    st.session_state.clinical_note_input = CKD_DEMO
if "nav" not in st.session_state:
    st.session_state.nav = NAV_HOME

# Programmatic page changes (buttons) are applied BEFORE the nav widget is created.
if st.session_state.get("goto"):
    st.session_state.nav = st.session_state.pop("goto")


def go(page: str):
    st.session_state.goto = page
    st.rerun()


# ============================================================
# 6. SIDEBAR: BRAND, NAVIGATION, STATUS, SETTINGS
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">🩺</div>'
        '<div><b>NephroCare AI</b><small>Kidney Disease Diagnostic System</small></div></div>',
        unsafe_allow_html=True,
    )
    st.radio("Navigation", NAV_ITEMS, key="nav", label_visibility="collapsed")
    st.divider()

    ml_model, model_err = load_ml_model(DEFAULT_MODEL_PATH)
    if ml_model is not None and getattr(ml_model, "scaler", None) is None:
        for sp in [DEFAULT_SCALER_PATH, FALLBACK_SCALER_PATH, "scaler.pkl", os.path.join(os.path.dirname(DEFAULT_MODEL_PATH), "scaler.pkl")]:
            if os.path.exists(sp):
                try:
                    ml_model.scaler = joblib.load(sp)
                    break
                except Exception:
                    pass
    env_groq_key = os.getenv("GROQ_API_KEY", "")

    if ml_model is not None:
        has_scaler = hasattr(ml_model, "scaler") and ml_model.scaler is not None
        if has_scaler:
            st.success("ML Kidney Model + Scaler: Active", icon="✅")
        else:
            st.warning("ML Model: Active (Scaler missing)", icon="⚠️")
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
        groq_api_key = st.text_input("Groq API Key", value=env_groq_key, type="password", placeholder="gsk_...")

    st.caption("💡 Tip: use the 1-click demo buttons on the New Screening page to test CKD vs. Healthy cases.")


# ============================================================
# 7. PIPELINE ANIMATION + REGISTRY TABLE HELPERS
# ============================================================
AI_FLOW = {
    "nodes": [
        ("📄", "Patient Data", "Clinical note", "src"),
        ("🤖", "LLM", "Scanning", "llm"),
        ("🌳", "ML Model", "Ensemble trees", "ml"),
        ("🤖", "LLM", "Summary", "llm"),
        ("📋", "Report", "Doctor + Patient", "out"),
    ],
    "stages": [
        ("send", 0, "📤", "Sending patient data to the LLM", "Your clinical note is on its way to the AI"),
        ("work", 1, "🔬", "LLM is scanning the note", "Finding patient details and all 20 kidney biomarkers"),
        ("send", 1, "📦", "Sending extracted data to the ML model", "Passing the structured lab values on for prediction"),
        ("work", 2, "🌳", "Ensemble decision trees are voting", "Many decision trees analyse the values and predict CKD or healthy"),
        ("send", 2, "📨", "Sending the prediction back to the LLM", "The result and lab values go to the AI for explanation"),
        ("work", 3, "✍️", "LLM is writing the summaries", "Drafting the doctor summary and a simple patient guide"),
        ("send", 3, "📋", "Preparing your report", "Putting everything together"),
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
        ("send", 0, "📤", "Sending your 20 lab inputs to the ML model", "Every biomarker is checked and encoded"),
        ("work", 1, "🌳", "Ensemble decision trees are voting", "Many decision trees analyse the values and predict CKD or healthy"),
        ("send", 1, "📨", "Sending the prediction to the LLM", "The result and lab values go to the AI for explanation"),
        ("work", 2, "✍️", "LLM is writing the summaries", "Drafting the doctor summary and a simple patient guide"),
        ("send", 2, "📋", "Preparing your report", "Putting everything together"),
    ],
}

KEY_CHIP_FEATS = ["sc", "bu", "hemo", "bp", "bgr", "al"]


def key_found_chips(data: dict) -> list:
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
    core_cls = "ai-core ai-core-done" if all_done else "ai-core"
    done_cls = " ai-think-done" if all_done else ""

    markup = f"""
    <div class="ai-think{done_cls}">
        <div class="flow-head">
            <div class="{core_cls}">{icon}</div>
            <div class="ai-body">
                <span class="pill-tag">✨ NephroCare AI pipeline</span>
                <h3 class="ai-title">{html.escape(title)}{dots}</h3>
                <p class="ai-detail">{html.escape(detail)}</p>
            </div>
            <div class="ai-pct">{pct}%</div>
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
    grp = (
        '<tr class="grp"><th colspan="3" class="g-report">📄 Report</th>'
        '<th colspan="2" class="g-pred">🎯 AI Prediction</th>'
        f'<th colspan="{len(FEATURES)}" class="g-data">🧪 Patient Data (20 inputs)</th></tr>'
    )
    cols = (
        '<tr class="cols"><th class="sticky-1">ID</th><th class="sticky-2">Patient</th><th>Date</th>'
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
        '<div class="reg-wrap"><div class="reg-head"><div>'
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
# 8. FULL DIAGNOSTIC REPORT (doctor + patient + 20 biomarkers + downloads)
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

    flagged_n = sum(1 for f in FEATURES if is_abnormal_biomarker(f, u_data.get(f, DEFAULTS[f])))
    theme = "ckd" if is_ckd else "ok"
    accent = "#E11D48" if is_ckd else "#059669"
    ring_deg = max(0.0, min(100.0, conf)) * 3.6

    if is_ckd:
        risk_text, headline = "⚠️ High Risk • Nephrology Attention Needed", "Chronic Kidney Disease (CKD) Detected"
    else:
        risk_text, headline = "✅ Low Risk • Normal Renal Function", "Healthy Kidney Profile (No CKD Detected)"
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

    hero = f"""
    <div class="sum-hero sum-hero-{theme}">
        <div class="conf-ring" style="background: conic-gradient({accent} {ring_deg:.1f}deg, #E2E8F0 0deg);">
            <div class="conf-ring-inner"><strong style="color:{accent};">{conf:.1f}%</strong><span>Confidence</span></div>
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

    doc_card = f"""
    <div class="sum-card sum-card-doc">
        <div class="sum-card-head">
            <div class="sum-icon sum-icon-doc">👨‍⚕️</div>
            <div><span class="pill-tag">Doctor's Side • Clinical Overview</span>
            <h3 class="card-title">Physician Assessment Summary</h3></div>
        </div>
        <p class="card-sub">Clear, natural clinical summary &amp; key diagnostic findings</p>
        <div>{doctor_html}</div>
    """
    if drivers_html:
        doc_card += f'<div class="sum-section"><div class="sum-section-title" style="color:#0369A1;">🎯 Main Clinical Drivers</div>{drivers_html}</div>'
    if abnormal_html:
        doc_card += f'<div class="sum-section sum-section-warn"><div class="sum-section-title" style="color:#BE123C;">🔬 Abnormal Biomarkers Flagged</div>{abnormal_html}</div>'
    doc_card += "</div>"

    pat_card = f"""
    <div class="sum-card sum-card-pat">
        <div class="sum-card-head">
            <div class="sum-icon sum-icon-pat">💙</div>
            <div><span class="pill-tag" style="background:#CCFBF1; color:#0F766E;">Patient's Side • Health Guide</span>
            <h3 class="card-title">What Your Results Mean for You</h3></div>
        </div>
        <p class="card-sub">Easy-to-understand explanation &amp; daily kidney care plan</p>
        <div>{patient_html}</div>
    """
    if actions_html:
        pat_card += f'<div class="sum-section sum-section-teal"><div class="sum-section-title" style="color:#0F766E;">✅ Recommended Next Steps &amp; Daily Care Plan</div>{actions_html}</div>'
    pat_card += "</div>"

    st.markdown(
        flat_html(f'<div class="summary-stage">{hero}<div class="sum-grid">{doc_card}{pat_card}</div></div>'),
        unsafe_allow_html=True,
    )

    tiles = (
        '<div class="bio-panel-box"><span class="pill-tag">🧪 Complete Lab Panel</span>'
        '<h3 class="card-title">All 20 Kidney Biomarkers & Reference Status</h3><div class="bio-grid-20">'
    )
    for feat in FEATURES:
        label, unit, ref = FEATURE_META[feat]
        raw = u_data.get(feat, DEFAULTS[feat])
        val = html.escape(str(raw))
        flagged = is_abnormal_biomarker(feat, raw)
        tile_cls = "bio-tile-flag" if flagged else "bio-tile-ok"
        badge_cls = "badge-flag" if flagged else "badge-ok"
        badge_txt = "⚠️ Abnormal" if flagged else "✓ Normal"
        val_color = "#BE123C" if flagged else "#0F172A"
        tiles += (
            f'<div class="{tile_cls}">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">'
            f'<span style="font-size:0.76rem; font-weight:700; color:#475569;">{label}</span>'
            f'<span class="{badge_cls}">{badge_txt}</span></div>'
            f'<div style="font-size:1.18rem; font-weight:800; color:{val_color};">{val} '
            f'<span style="font-size:0.75rem; font-weight:500; color:#64748B;">{unit}</span></div>'
            f'<div style="font-size:0.72rem; color:#94A3B8; margin-top:0.2rem;">Normal Ref: {ref}</div></div>'
        )
    tiles += "</div></div>"
    st.markdown(flat_html(tiles), unsafe_allow_html=True)

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
# 9. SCREENING POPUP (live pipeline animation)
# ============================================================
SEND_PAUSE = 0.3
ML_PAUSE = 0.5


def _screening_cta():
    st.markdown('<div class="cta-note">✅ Analysis complete! Check your summary and records.</div>', unsafe_allow_html=True)
    if st.button("📊 Open Dashboard & Report  →", type="primary", use_container_width=True, key="cta_go_patient"):
        go(NAV_DASH)


def _make_summaries(patients: list, preds: list, api_key, model_name) -> list:
    _, _, chain = get_llm_chains(api_key, model_name)

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
    flow = AI_FLOW if mode == "ai" else MANUAL_FLOW
    n = len(flow["stages"])
    done_key = f"job_done_{job_id}"
    slot = st.empty()

    if st.session_state.get(done_key):
        show_thinking(slot, flow, n, found=st.session_state.get("last_chips"))
        _screening_cta()
        return

    step = "Extraction" if mode == "ai" else "Prediction"
    try:
        if mode == "ai":
            show_thinking(slot, flow, 0)
            time.sleep(SEND_PAUSE)

            show_thinking(slot, flow, 1)
            patients = extract_from_text(data, api_key, model_name)
            chips = key_found_chips(patients[0])
            if len(patients) > 1:
                chips.insert(0, ("Patients found", len(patients), False))
            st.session_state.extracted_raw = patients[0]
            st.session_state.patient_form_data = sanitize_user_data(patients[0])

            show_thinking(slot, flow, 2, found=chips)
            time.sleep(SEND_PAUSE)

            step = "Prediction"
            show_thinking(slot, flow, 3, found=chips)
            preds = [predict(p, ml_model) for p in patients]
            time.sleep(ML_PAUSE)

            show_thinking(slot, flow, 4, found=chips)
            time.sleep(SEND_PAUSE)

            step = "Summary"
            show_thinking(slot, flow, 5, found=chips)
            summaries = _make_summaries(patients, preds, api_key, model_name)

            show_thinking(slot, flow, 6, found=chips)
            entries = [save_record(p, pr, cf, sm) for p, (pr, cf), sm in zip(patients, preds, summaries)]
        else:
            chips = key_found_chips(data)
            show_thinking(slot, flow, 0, found=chips)
            time.sleep(SEND_PAUSE)

            show_thinking(slot, flow, 1, found=chips)
            pred, confidence = predict(data, ml_model)
            time.sleep(ML_PAUSE)

            show_thinking(slot, flow, 2, found=chips)
            time.sleep(SEND_PAUSE)

            step = "Summary"
            show_thinking(slot, flow, 3, found=chips)
            if api_key:
                summary = _make_summaries([data], [(pred, confidence)], api_key, model_name)[0]
            else:
                summary = ("**Doctor:** Groq API key not configured.\n\n"
                           "**Patient:** Please add your Groq API key to generate the patient guide.")

            show_thinking(slot, flow, 4, found=chips)
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
    st.toast(f"{len(entries)} report{'s' if len(entries) != 1 else ''} saved!", icon="✅")
    _screening_cta()


# ============================================================
# 10. PAGES
# ============================================================
KIDNEY_SVG = """
<svg viewBox="0 0 240 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="kg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FDA4AF"/><stop offset="1" stop-color="#E11D48"/>
    </linearGradient>
    <linearGradient id="vg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#38BDF8"/><stop offset="1" stop-color="#0284C7"/>
    </linearGradient>
  </defs>
  <path d="M95 22 C50 14 18 52 24 100 C30 150 78 176 98 148 C110 130 82 122 86 100 C90 80 118 72 114 48 C112 34 106 24 95 22Z" fill="url(#kg)"/>
  <path d="M145 22 C190 14 222 52 216 100 C210 150 162 176 142 148 C130 130 158 122 154 100 C150 80 122 72 126 48 C128 34 134 24 145 22Z" fill="url(#kg)"/>
  <path d="M120 10 L120 70 M120 70 C120 90 100 100 96 118 M120 70 C120 90 140 100 144 118" stroke="url(#vg)" stroke-width="7" stroke-linecap="round" fill="none"/>
  <path d="M96 118 L96 186 M144 118 L144 186" stroke="#FBBF24" stroke-width="6" stroke-linecap="round" fill="none"/>
  <circle cx="60" cy="90" r="5" fill="#FFF" opacity="0.35"/><circle cx="180" cy="90" r="5" fill="#FFF" opacity="0.35"/>
</svg>
"""


def page_home():
    st.markdown(
        flat_html(f"""
        <div class="home-hero">
            <div>
                <span class="pill-tag">🏥 Clinical Decision Support</span>
                <h1>NephroCare AI</h1>
                <h3>Kidney Disease Diagnostic Prediction System</h3>
                <p><b>Early Detection. Better Care.</b></p>
                <p>Paste a lab note or enter 20 kidney biomarkers. Our AI extracts the data, an ML ensemble predicts CKD risk,
                and you get a clear report for the doctor and a simple guide for the patient.</p>
            </div>
            <div class="home-art">{KIDNEY_SVG}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Start Screening  →", type="primary", use_container_width=True, key="hero_cta"):
            go(NAV_SCREEN)
    with c2:
        if st.button("📊 View Dashboard", use_container_width=True, key="home_dash"):
            go(NAV_DASH)
    with c3:
        if st.button("📁 Saved Reports", use_container_width=True, key="home_reports"):
            go(NAV_REPORTS)

    st.markdown(
        flat_html("""
        <div class="feat-row">
            <div class="feat-card"><div class="fi">📄</div><b>AI Lab Note Extraction</b><span>Paste text or upload .txt / .csv / .json. One or many patients per file.</span></div>
            <div class="feat-card"><div class="fi">🌳</div><b>ML CKD Prediction</b><span>Ensemble decision trees classify CKD vs. healthy with a confidence score.</span></div>
            <div class="feat-card"><div class="fi">👨‍⚕️</div><b>Doctor + Patient Reports</b><span>A clinical overview and a plain-language health guide, ready to download.</span></div>
            <div class="feat-card"><div class="fi">🗂️</div><b>Patient Registry</b><span>Every screening saved with all 20 inputs. Search, filter and export.</span></div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def _render_cohort_triage_board(batch: list, ml_model):
    annotated = []
    for r in batch:
        tinfo = assign_triage_tier(r)
        annotated.append((r, tinfo))

    total_count = len(annotated)
    urgent_count = sum(1 for _, t in annotated if t["tier_key"] == "urgent")
    monitor_count = sum(1 for _, t in annotated if t["tier_key"] == "monitor")
    routine_count = sum(1 for _, t in annotated if t["tier_key"] == "routine")

    urgent_pct = (urgent_count / total_count * 100) if total_count else 0
    monitor_pct = (monitor_count / total_count * 100) if total_count else 0
    routine_pct = (routine_count / total_count * 100) if total_count else 0

    st.markdown(
        flat_html(f"""
        <div class="triage-hero-box">
            <span class="triage-meta-pill">⛺ Mobile Health Mission #04 • Rural Sub-District Health Outpost</span>
            <h2>Field Triage Assistant & Cohort Report Cards</h2>
            <p>Population stratification protocol for rural and remote screening drives. Screened individuals are automatically categorized into actionable clinical tiers based on renal biomarkers and ML ensemble risk index.</p>
            <div class="triage-hero-meta">
                <span class="triage-meta-pill">👨‍⚕️ Triage Officer: On-Duty Medical Officer</span>
                <span class="triage-meta-pill">👥 Cohort Size: {total_count} Screened</span>
                <span class="triage-meta-pill">⚡ Engine: Hybrid Ensemble + Clinical Rules</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown(
        flat_html(f"""
        <div class="triage-kpi-grid">
            <div class="triage-kpi-card total">
                <div class="tkpi-header"><span class="tkpi-title">Total Cohort</span><span>👥</span></div>
                <div class="tkpi-val total">{total_count}</div>
                <div class="tkpi-sub">Field Patients Screened</div>
            </div>
            <div class="triage-kpi-card urgent">
                <div class="tkpi-header"><span class="tkpi-title">🔴 Urgent Referral</span><span>🚨</span></div>
                <div class="tkpi-val urgent">{urgent_count}</div>
                <div class="tkpi-sub">{urgent_pct:.0f}% • Tertiary Hospital Transfer</div>
            </div>
            <div class="triage-kpi-card monitor">
                <div class="tkpi-header"><span class="tkpi-title">🟡 Priority Monitor</span><span>⚠️</span></div>
                <div class="tkpi-val monitor">{monitor_count}</div>
                <div class="tkpi-sub">{monitor_pct:.0f}% • Primary Care Observation</div>
            </div>
            <div class="triage-kpi-card routine">
                <div class="tkpi-header"><span class="tkpi-title">🟢 Routine Wellness</span><span>✅</span></div>
                <div class="tkpi-val routine">{routine_count}</div>
                <div class="tkpi-sub">{routine_pct:.0f}% • Normal Baseline Cleared</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Interactive Tier Filter
    filter_choice = st.radio(
        "Triage Filter",
        options=[
            f"All Patients ({total_count})",
            f"🔴 Urgent Referrals ({urgent_count})",
            f"🟡 Priority Monitor ({monitor_count})",
            f"🟢 Routine Wellness ({routine_count})"
        ],
        horizontal=True,
        key="triage_filter_radio",
        label_visibility="collapsed"
    )

    # Sort annotated: Urgent (rank 1), Monitor (rank 2), Routine (rank 3), then conf desc
    sorted_cohort = sorted(
        annotated,
        key=lambda item: (item[1]["tier_rank"], -float(item[0].get("prediction", {}).get("confidence", 0)))
    )

    if filter_choice.startswith("🔴"):
        active_cohort = [item for item in sorted_cohort if item[1]["tier_key"] == "urgent"]
    elif filter_choice.startswith("🟡"):
        active_cohort = [item for item in sorted_cohort if item[1]["tier_key"] == "monitor"]
    elif filter_choice.startswith("🟢"):
        active_cohort = [item for item in sorted_cohort if item[1]["tier_key"] == "routine"]
    else:
        active_cohort = sorted_cohort

    st.markdown(f'<div style="margin: 0.8rem 0 1.2rem 0; font-weight:800; color:#334155; font-size:0.92rem;">Showing {len(active_cohort)} Patient Report Card{"s" if len(active_cohort) != 1 else ""} sorted by clinical severity:</div>', unsafe_allow_html=True)

    # Render Report Cards in 2 columns
    col_left, col_right = st.columns(2, gap="large")
    for idx, (entry, tinfo) in enumerate(active_cohort):
        target_col = col_left if idx % 2 == 0 else col_right
        ud = sanitize_user_data(entry.get("user_data", {}))
        pr = entry.get("prediction", {})
        is_ckd = pr.get("value") == 1
        conf = float(pr.get("confidence", 0))

        pname = html.escape(str(ud.get("name") or f"Patient #{idx+1}"))
        page = html.escape(str(ud.get("age", "—")))
        pid = html.escape(str(entry.get("id", idx+1)))

        tier_k = tinfo["tier_key"]
        tier_lbl = tinfo["tier_lbl"]
        b_icon = tinfo["badge_icon"]

        pred_title = "Chronic Kidney Disease (CKD)" if is_ckd else "Healthy Baseline (No CKD)"

        # Mini biomarker flags
        sc_val = ud.get("sc")
        sc_str = f"{sc_val} mg/dL" if sc_val is not None else "—"
        sc_abn = is_abnormal_biomarker("sc", sc_val)

        bu_val = ud.get("bu")
        bu_str = f"{bu_val} mg/dL" if bu_val is not None else "—"
        bu_abn = is_abnormal_biomarker("bu", bu_val)

        al_val = ud.get("al")
        al_str = f"Grade {al_val}" if al_val is not None else "—"
        al_abn = is_abnormal_biomarker("al", al_val)

        hemo_val = ud.get("hemo")
        hemo_str = f"{hemo_val} g/dL" if hemo_val is not None else "—"
        hemo_abn = is_abnormal_biomarker("hemo", hemo_val)

        bp_val = ud.get("bp")
        bp_str = f"{bp_val} mm/Hg" if bp_val is not None else "—"
        bp_abn = is_abnormal_biomarker("bp", bp_val)

        bgr_val = ud.get("bgr")
        bgr_str = f"{bgr_val} mg/dL" if bgr_val is not None else "—"
        bgr_abn = is_abnormal_biomarker("bgr", bgr_val)

        card_html = f"""
        <div class="triage-report-card {tier_k}">
            <div>
                <div class="trc-header">
                    <div class="trc-patient-info">
                        <div class="trc-avatar {tier_k}">{b_icon}</div>
                        <div>
                            <div class="trc-name">{pname}</div>
                            <div class="trc-meta-row">
                                <span class="trc-age-badge">{page} yrs</span>
                                <span class="trc-id-badge">ID: #{pid}</span>
                            </div>
                        </div>
                    </div>
                    <span class="trc-tier-badge {tier_k}">{b_icon} {tier_lbl}</span>
                </div>
                <div class="trc-conf-strip {tier_k}">
                    <span class="pred-lbl">{pred_title}</span>
                    <span class="conf-val">{conf:.1f}% Confidence</span>
                </div>
                <div class="trc-bio-matrix">
                    <div class="trc-bio-pill {'abnormal' if sc_abn else ''}">
                        <span class="trc-bio-lbl">Creatinine</span>
                        <span class="trc-bio-val">{sc_str}</span>
                    </div>
                    <div class="trc-bio-pill {'abnormal' if bu_abn else ''}">
                        <span class="trc-bio-lbl">Blood Urea</span>
                        <span class="trc-bio-val">{bu_str}</span>
                    </div>
                    <div class="trc-bio-pill {'abnormal' if al_abn else ''}">
                        <span class="trc-bio-lbl">Albumin</span>
                        <span class="trc-bio-val">{al_str}</span>
                    </div>
                    <div class="trc-bio-pill {'abnormal' if hemo_abn else ''}">
                        <span class="trc-bio-lbl">Hemoglobin</span>
                        <span class="trc-bio-val">{hemo_str}</span>
                    </div>
                    <div class="trc-bio-pill {'abnormal' if bp_abn else ''}">
                        <span class="trc-bio-lbl">Blood Press.</span>
                        <span class="trc-bio-val">{bp_str}</span>
                    </div>
                    <div class="trc-bio-pill {'abnormal' if bgr_abn else ''}">
                        <span class="trc-bio-lbl">Glucose</span>
                        <span class="trc-bio-val">{bgr_str}</span>
                    </div>
                </div>
                <div class="trc-directive-box {tier_k}">
                    <span class="trc-directive-icon">{b_icon}</span>
                    <div class="trc-directive-text">
                        <b>{tinfo['directive_title']}</b>
                        {tinfo['directive']}
                    </div>
                </div>
            </div>
        </div>
        """
        with target_col:
            st.markdown(flat_html(card_html), unsafe_allow_html=True)
            with st.expander(f"📋 View Full Diagnostic Report for {pname}", expanded=False):
                render_diagnostic_report(entry, key_prefix=f"triage_rep_{entry['id']}_{idx}")

    # Roster Export
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">📥 Export Camp Triage Manifest</h3>', unsafe_allow_html=True)
    roster_rows = []
    for r, t in sorted_cohort:
        ud = r.get("user_data", {})
        pr = r.get("prediction", {})
        roster_rows.append({
            "Camp_Patient_ID": r.get("id"),
            "Patient_Name": ud.get("name"),
            "Age": ud.get("age"),
            "Triage_Tier": t["tier_lbl"],
            "Clinical_Directive": t["directive"],
            "Diagnosis": "CKD Detected" if pr.get("value") == 1 else "Healthy Baseline",
            "Confidence_Pct": round(float(pr.get("confidence", 0)), 1),
            "Serum_Creatinine_mg_dL": ud.get("sc"),
            "Blood_Urea_mg_dL": ud.get("bu"),
            "Urine_Albumin_Grade": ud.get("al"),
            "Hemoglobin_g_dL": ud.get("hemo"),
            "Blood_Pressure_mmHg": ud.get("bp"),
            "Blood_Glucose_mg_dL": ud.get("bgr"),
            "Hypertension": ud.get("htn"),
            "Diabetes": ud.get("dm"),
            "Timestamp": r.get("timestamp")
        })
    roster_df = pd.DataFrame(roster_rows)
    csv_bytes = roster_df.to_csv(index=False).encode("utf-8")
    
    col_exp1, col_exp2 = st.columns([1.5, 2.5])
    with col_exp1:
        st.download_button(
            label="📥 Download Rural Camp Triage Roster (CSV)",
            data=csv_bytes,
            file_name="rural_health_camp_triage_roster.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        st.caption("Includes all 10 triaged patients, color-coded priority levels, clinical action orders, and complete biomarker findings.")


def page_dashboard():
    st.markdown('<h1 class="page-title">Screening Dashboard</h1><p class="page-sub">Overview of all screenings, rural camp cohort triage, and individual diagnostic reports.</p>', unsafe_allow_html=True)

    records = load_all_records()
    total = len(records)
    ckd = sum(1 for r in records if r.get("prediction", {}).get("value") == 1)
    healthy = total - ckd
    avg_conf = (sum(r.get("prediction", {}).get("confidence", 0) for r in records) / total) if total else 0

    st.markdown(
        flat_html(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-ico blue">🧪</div><div><div class="stat-lbl">Total Scans</div><div class="stat-val">{total}</div></div></div>
            <div class="stat-card"><div class="stat-ico green">✅</div><div><div class="stat-lbl">Healthy</div><div class="stat-val green">{healthy}</div></div></div>
            <div class="stat-card"><div class="stat-ico red">⚠️</div><div><div class="stat-lbl">CKD Detected</div><div class="stat-val red">{ckd}</div></div></div>
            <div class="stat-card"><div class="stat-ico teal">🎯</div><div><div class="stat-lbl">Avg Confidence</div><div class="stat-val blue">{avg_conf:.1f}%</div></div></div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Action Box Launcher: Rural Camp Triage Mode
    st.markdown(
        flat_html("""
        <div class="triage-action-box">
            <div class="tab-content">
                <span class="tab-tag">⛺ Rural Health Camp Assistant</span>
                <h3 class="tab-title">🏥 Cohort Triage Board & Report Cards</h3>
                <p class="tab-desc">Simulate a field screening mission in an underserved rural outpost. Evaluates a 10-patient cohort simultaneously and stratifies them into <b>🔴 Urgent Referral</b>, <b>🟡 Priority Monitor</b>, and <b>🟢 Routine Wellness</b> with field action directives.</p>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )
    col_act1, col_act2 = st.columns([1.5, 2.5])
    with col_act1:
        if st.button("⛺ Launch 10-Patient Rural Camp Triage Board", type="primary", use_container_width=True, key="dash_launch_camp"):
            if ml_model is not None:
                camp_batch = load_rural_camp_demo_cohort(ml_model)
                st.session_state.latest_batch = camp_batch
                st.session_state.latest_entry = camp_batch[0]
                st.session_state.dash_view_mode = "⛺ Cohort Triage Board (Report Cards)"
                st.toast("⛺ Rural Camp Cohort Loaded (10 Patients Triaged)!", icon="✅")
                st.rerun()
            else:
                st.error("ML Model is not loaded.")
    with col_act2:
        st.caption("1-click loads 10 real rural camp clinical cases, runs ML inference, and generates the color-coded triage board.")

    batch = st.session_state.get("latest_batch") or ([st.session_state.latest_entry] if st.session_state.latest_entry else [])
    if not batch:
        st.markdown(
            flat_html("""
            <div class="card" style="text-align:center; padding:2.4rem;">
                <h3 class="card-title">No individual screening yet</h3>
                <p class="card-sub">Run a new screening or launch the 10-patient rural health camp triage board above.</p>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("🩺 Start a New Screening", type="primary", key="dash_start"):
            go(NAV_SCREEN)
        return

    # If batch has multiple patients, provide view switcher
    if len(batch) > 1:
        dash_mode = st.radio(
            "Dashboard View Mode",
            options=["⛺ Cohort Triage Board (Report Cards)", "👤 Individual Patient Detailed Diagnostic"],
            horizontal=True,
            key="dash_view_mode",
            label_visibility="collapsed"
        )
    else:
        dash_mode = "👤 Individual Patient Detailed Diagnostic"

    if dash_mode.startswith("⛺"):
        _render_cohort_triage_board(batch, ml_model)
        return

    # Otherwise render Individual Patient view
    entry = batch[0]
    if len(batch) > 1:
        st.markdown(f'<span class="pill-tag">👥 {len(batch)} patients found in this upload</span>', unsafe_allow_html=True)
        options = list(range(len(batch)))

        def _lbl(i):
            b = batch[i]
            nm = str(b.get("user_data", {}).get("name") or "Anonymous Patient")
            return f"{nm} — {b['prediction']['label']} ({b['prediction']['confidence']:.1f}%)"

        pick = st.radio("Patients in this upload", options, format_func=_lbl, horizontal=True,
                        key="batch_pick", label_visibility="collapsed")
        entry = batch[pick]

    pred = entry["prediction"]
    is_ckd = pred["value"] == 1
    conf = float(pred["confidence"])
    cls = "ckd" if is_ckd else "ok"
    big = "CKD Detected" if is_ckd else "Healthy"
    icon = "⚠️" if is_ckd else "✅"
    parsed = parse_summary_sections(entry.get("summary", ""))
    short = html.escape(to_single_paragraph(parsed["doctor"] or entry.get("summary", ""), 3))
    pname = html.escape(str(entry.get("user_data", {}).get("name") or "Anonymous Patient"))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            flat_html(f"""
            <div class="card">
                <span class="pill-tag">Prediction Result</span>
                <h3 class="card-title">👤 {pname}</h3>
                <div class="pred-box {cls}"><div style="font-size:1.8rem;">{icon}</div><h2>{big}</h2><p>Confidence: {conf:.1f}%</p></div>
                <div class="conf-bar"><div class="{cls}" style="width:{max(0.0, min(100.0, conf)):.1f}%;"></div></div>
            </div>
            """),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            flat_html(f"""
            <div class="card">
                <span class="pill-tag">Clinical Summary</span>
                <h3 class="card-title">Doctor's Overview</h3>
                <p class="readable-paragraph">{short}</p>
                <div class="conf-bar"><div class="{cls}" style="width:{max(0.0, min(100.0, conf)):.1f}%;"></div></div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    render_diagnostic_report(entry, key_prefix=f"dash{entry['id']}")

    name = str(entry.get("user_data", {}).get("name") or "Anonymous Patient")
    if len(batch) > 1:
        recs, caption = list(reversed(batch)), f"All {len(batch)} patients from this upload"
    else:
        recs = [r for r in reversed(records) if str(r.get("user_data", {}).get("name") or "Anonymous Patient") == name]
        caption = f"Saved records for {name}"
    if recs:
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown(flat_html(build_registry_table(recs)), unsafe_allow_html=True)
        st.caption(f"{caption} • Red cells = abnormal values • Scroll sideways to see all 20 inputs")


def page_screening():
    st.markdown('<h1 class="page-title">New Screening</h1><p class="page-sub">Paste a lab note, upload a report, or enter all 20 biomarkers manually.</p>', unsafe_allow_html=True)
    tab_ai, tab_manual = st.tabs(["📋 AI Lab Note Screening", "🎛️ Manual Lab Calculator (20 Markers)"])

    # ---------------- TAB: AI LAB NOTE ----------------
    with tab_ai:
        col_in, col_snap = st.columns([1.0, 1.0], gap="large")
        raw_clinical_text = ""
        with col_in:
            with st.container(border=True):
                st.markdown('<span class="pill-tag">Step 1 • Patient Input</span>', unsafe_allow_html=True)
                st.markdown('<h3 class="card-title">📥 Enter or Upload Patient Lab Information</h3>', unsafe_allow_html=True)
                st.markdown('<p class="card-sub">Choose a demo patient, paste a clinical paragraph, or upload a lab text file.</p>', unsafe_allow_html=True)

                intake_mode = st.radio(
                    "Input Method",
                    options=["📝 Paste Clinical Note / Lab Text", "📂 Upload Lab Report File (.txt, .json, .csv)"],
                    horizontal=True, label_visibility="collapsed", key="intake_mode",
                )

                if intake_mode.startswith("📝"):
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("🔴 CKD Demo (Ahmed, 62)", use_container_width=True):
                            st.session_state.clinical_note_input = CKD_DEMO
                    with b2:
                        if st.button("🟢 Healthy Demo (Sara, 34)", use_container_width=True):
                            st.session_state.clinical_note_input = HEALTHY_DEMO
                    with b3:
                        if st.button("⛺ 10-Patient Camp Demo", use_container_width=True):
                            st.session_state.clinical_note_input = RURAL_CAMP_10_TEXT
                            st.toast("⛺ Loaded 10-Patient Rural Camp Roster!", icon="⛺")
                    raw_clinical_text = st.text_area(
                        "Patient Lab Note / Clinical Paragraph",
                        key="clinical_note_input", height=180,
                        placeholder="Type or paste patient lab findings here...",
                    )
                else:
                    uploaded_report = st.file_uploader("Upload Patient Report File (.txt, .csv, .json)", type=["txt", "md", "csv", "json"])
                    if uploaded_report is not None:
                        raw_clinical_text = uploaded_report.read().decode("utf-8", errors="ignore")
                        st.text_area("Uploaded File Preview", value=raw_clinical_text, height=150, disabled=True)

                run_btn = st.button("🩺 Run Kidney Screening & Generate Doctor + Patient Report", type="primary", use_container_width=True)

        if run_btn:
            if not raw_clinical_text.strip():
                st.error("Please enter a clinical note or upload a file first.")
            elif not groq_api_key:
                st.error("Please enter your GROQ_API_KEY in `.env` or the sidebar settings.")
            elif ml_model is None:
                st.error("ML Model (`kidney_model.pkl`) is not loaded. Check the sidebar settings.")
            else:
                run_screening_dialog("ai", raw_clinical_text, ml_model, groq_api_key, llm_model_name, str(time.time()))

        with col_snap:
            if st.session_state.extracted_raw is not None:
                ext = st.session_state.extracted_raw
                clean_ext = sanitize_user_data(ext)
                found_count = sum(1 for k in FEATURES if ext.get(k) is not None)
                p_name = html.escape(str(clean_ext.get("name") or "Anonymous Patient"))
                p_age = html.escape(str(clean_ext.get("age", "—")))

                FEATURE_ICONS = {
                    "sc": "🧪",
                    "bu": "💧",
                    "hemo": "🩸",
                    "bp": "💓",
                    "bgr": "⚡",
                    "al": "🔬",
                    "htn": "🫀",
                    "dm": "🩺",
                }
                DISPLAY_NAMES = {
                    "sc": "Serum Creatinine",
                    "bu": "Blood Urea",
                    "hemo": "Hemoglobin",
                    "bp": "Blood Pressure",
                    "bgr": "Blood Glucose",
                    "al": "Urine Albumin",
                    "htn": "Hypertension",
                    "dm": "Diabetes Mellitus",
                }
                cards_html = '<div class="bio-metric-grid">'
                flagged_count = 0
                for k in ["sc", "bu", "hemo", "bp", "bgr", "al", "htn", "dm"]:
                    lbl, unit, ref = FEATURE_META[k]
                    disp_name = DISPLAY_NAMES.get(k, lbl)
                    icon = FEATURE_ICONS.get(k, "🔬")
                    raw_val = clean_ext.get(k)
                    flag = is_abnormal_biomarker(k, raw_val)
                    if flag:
                        flagged_count += 1

                    if raw_val is None:
                        val_str = "—"
                        val_display_unit = unit if unit not in ("history", "scale 0–5") else ""
                    elif k in ("htn", "dm"):
                        val_str = "Yes" if str(raw_val).lower() == "yes" else "No"
                        val_display_unit = "History"
                    elif k == "al":
                        val_str = f"Grade {raw_val}"
                        val_display_unit = "(scale 0–5)"
                    else:
                        val_str = html.escape(str(raw_val))
                        val_display_unit = unit if unit not in ("history", "scale 0–5") else ""

                    card_cls = "flag" if flag else "ok"
                    badge_cls = "flag" if flag else "ok"
                    badge_txt = "⚠️ Abnormal" if flag else "✓ Normal"

                    unit_span = f'<span class="bmc-unit">{html.escape(val_display_unit)}</span>' if val_display_unit else ""
                    ref_text = html.escape(str(ref))
                    if k in ("htn", "dm"):
                        ref_text = "No history"
                    elif k == "al":
                        ref_text = "Grade 0"

                    cards_html += f"""
                    <div class="bio-metric-card {card_cls}">
                        <div class="bmc-top">
                            <div class="bmc-label-wrap">
                                <span class="bmc-icon">{icon}</span>
                                <span class="bmc-label" title="{html.escape(lbl)}">{html.escape(disp_name)}</span>
                            </div>
                            <span class="bmc-badge {badge_cls}">{badge_txt}</span>
                        </div>
                        <div class="bmc-middle">
                            <span class="bmc-val {card_cls}">{val_str}</span>
                            {unit_span}
                        </div>
                        <div class="bmc-bottom">
                            <div class="bmc-ref-tag">
                                <span class="ref-lbl">Ref</span>
                                <span class="ref-num">{ref_text}</span>
                            </div>
                        </div>
                    </div>
                    """
                cards_html += "</div>"

                footer_cls = "has-abnormal" if flagged_count > 0 else "all-normal"
                footer_icon = "⚠️" if flagged_count > 0 else "✅"
                footer_text = f"{footer_icon} {flagged_count} of 8 key renal markers flagged abnormal" if flagged_count > 0 else f"{footer_icon} All 8 key renal markers within normal reference ranges"
                footer_pill = "High Risk Detected" if flagged_count > 0 else "Baseline Verified"

                st.markdown(flat_html(f"""
                <div class="snapshot-card">
                    <div>
                        <span class="pill-tag">Step 2 • AI Extracted Profile</span>
                        <h3 class="card-title">🔬 Extracted Patient & Key Renal Markers</h3>
                        <p class="card-sub">Structured from the clinical note and ready for ML ensemble classification.</p>
                    </div>
                    <div class="patient-banner-v2">
                        <div class="pb-left">
                            <div class="pb-avatar">👤</div>
                            <div>
                                <div class="pb-eyebrow">Patient Profile</div>
                                <div class="pb-name">{p_name} <span class="pb-age-chip">{p_age} yrs</span></div>
                            </div>
                        </div>
                        <div class="pb-stat-badge">
                            <span class="pb-stat-lbl">Biomarkers Detected</span>
                            <div class="pb-stat-val"><b>{found_count}</b><span>/ 20 Extracted</span></div>
                        </div>
                    </div>
                    {cards_html}
                    <div class="snapshot-footer-bar {footer_cls}">
                        <span>{footer_text}</span>
                        <span class="sf-status-pill">{footer_pill}</span>
                    </div>
                </div>
                """), unsafe_allow_html=True)
            else:
                # Sleek awaiting state
                FEATURE_ICONS = {
                    "sc": "🧪",
                    "bu": "💧",
                    "hemo": "🩸",
                    "bp": "💓",
                    "bgr": "⚡",
                    "al": "🔬",
                    "htn": "🫀",
                    "dm": "🩺",
                }
                preview_feats = [
                    ("sc", "Serum Creatinine", "mg/dL", "0.6–1.3 mg/dL"),
                    ("bu", "Blood Urea", "mg/dL", "15–40 mg/dL"),
                    ("hemo", "Hemoglobin", "g/dL", "12.0–17.0 g/dL"),
                    ("bp", "Blood Pressure", "mm/Hg", "70–85 mm/Hg"),
                    ("bgr", "Blood Glucose", "mg/dL", "70–140 mg/dL"),
                    ("al", "Urine Albumin", "scale 0–5", "Grade 0"),
                    ("htn", "Hypertension", "history", "No history"),
                    ("dm", "Diabetes Mellitus", "history", "No history"),
                ]
                ghost_cards = '<div class="bio-metric-grid">'
                for fkey, plbl, punit, pref in preview_feats:
                    icon = FEATURE_ICONS.get(fkey, "🔬")
                    u_display = f'<span class="bmc-unit">{punit}</span>' if punit not in ("history", "scale 0–5") else ""
                    ghost_cards += f"""
                    <div class="bio-metric-card" style="opacity:0.82; border:1.5px dashed #CBD5E1; background:linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);">
                        <div class="bmc-top">
                            <div class="bmc-label-wrap">
                                <span class="bmc-icon" style="background:#F1F5F9;">{icon}</span>
                                <span class="bmc-label" style="color:#64748B;">{plbl}</span>
                            </div>
                            <span class="bmc-badge" style="background:#F1F5F9; color:#94A3B8; border:1px solid #E2E8F0;">Standby</span>
                        </div>
                        <div class="bmc-middle">
                            <span class="bmc-val" style="color:#94A3B8;">—</span>
                            {u_display}
                        </div>
                        <div class="bmc-bottom">
                            <div class="bmc-ref-tag">
                                <span class="ref-lbl">Ref</span>
                                <span class="ref-num">{pref}</span>
                            </div>
                        </div>
                    </div>
                    """
                ghost_cards += "</div>"

                st.markdown(flat_html(f"""
                <div class="snapshot-card">
                    <div>
                        <span class="pill-tag">Step 2 • AI Extracted Profile</span>
                        <h3 class="card-title">🔬 Extracted Patient & Key Renal Markers</h3>
                        <p class="card-sub">Your structured patient profile and highlighted lab values will appear here automatically.</p>
                    </div>
                    <div style="background:linear-gradient(135deg, #F0F9FF 0%, #E6F7F4 100%); border:1.5px dashed #BAE6FD; border-radius:18px; padding:0.95rem 1.25rem; margin:0.8rem 0 1.1rem 0; display:flex; align-items:center; gap:0.9rem;">
                        <div style="width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg, #0284C7, #0D9488); display:flex; align-items:center; justify-content:center; font-size:1.25rem; color:#FFFFFF; flex-shrink:0; box-shadow:0 4px 10px -2px rgba(2,132,199,0.35);">⚡</div>
                        <div>
                            <div style="font-size:0.72rem; font-weight:800; color:#0369A1; text-transform:uppercase; letter-spacing:0.06em;">Awaiting Clinical Input</div>
                            <div style="font-size:0.86rem; font-weight:600; color:#1E293B;">Select a demo patient or paste clinical notes on the left, then click <strong>Run Kidney Screening</strong>.</div>
                        </div>
                    </div>
                    {ghost_cards}
                    <div class="snapshot-footer-bar" style="background:#F8FAFC; border:1.5px solid #E2E8F0; color:#64748B;">
                        <span>⚡ 20 Renal biomarkers will be structured & validated automatically</span>
                        <span class="sf-status-pill" style="color:#0284C7; background:#E0F2FE;">Standby</span>
                    </div>
                </div>
                """), unsafe_allow_html=True)


    # ---------------- TAB: MANUAL ----------------
    with tab_manual:
        with st.container(border=True):
            st.markdown('<span class="pill-tag">Interactive Lab Calculator</span>', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title">🎛️ Enter or Edit All 20 Kidney Biomarkers</h3>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Modify any value to test how specific lab markers affect the CKD prediction and guidance.</p>', unsafe_allow_html=True)

            cur = st.session_state.patient_form_data

            def _idx(opts, key, default):
                v = str(cur.get(key, default)).lower()
                return opts.index(v if v in opts else default)

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
                rbc_opts, pcc_opts = ["normal", "abnormal"], ["notpresent", "present"]
                with u1:
                    f_al = st.selectbox("Albumin (0–5)", options=[0, 1, 2, 3, 4, 5], index=int(float(cur.get("al", 0))))
                with u2:
                    f_su = st.selectbox("Sugar (0–5)", options=[0, 1, 2, 3, 4, 5], index=int(float(cur.get("su", 0))))
                with u3:
                    f_rbc = st.selectbox("Red Blood Cells", options=rbc_opts, index=_idx(rbc_opts, "rbc", "normal"))
                with u4:
                    f_pc = st.selectbox("Pus Cells", options=rbc_opts, index=_idx(rbc_opts, "pc", "normal"))
                with u5:
                    f_pcc = st.selectbox("Pus Cell Clumps", options=pcc_opts, index=_idx(pcc_opts, "pcc", "notpresent"))
                with u6:
                    f_ba = st.selectbox("Bacteria", options=pcc_opts, index=_idx(pcc_opts, "ba", "notpresent"))

                st.markdown("##### 🩺 Medical History & Symptoms")
                h1, h2, h3, h4, h5, h6 = st.columns(6)
                yn_opts, ap_opts = ["no", "yes"], ["good", "poor"]
                with h1:
                    f_htn = st.selectbox("Hypertension", options=yn_opts, index=_idx(yn_opts, "htn", "no"))
                with h2:
                    f_dm = st.selectbox("Diabetes Mellitus", options=yn_opts, index=_idx(yn_opts, "dm", "no"))
                with h3:
                    f_cad = st.selectbox("Coronary Artery Dis.", options=yn_opts, index=_idx(yn_opts, "cad", "no"))
                with h4:
                    f_appet = st.selectbox("Appetite", options=ap_opts, index=_idx(ap_opts, "appet", "good"))
                with h5:
                    f_pe = st.selectbox("Pedal Edema", options=yn_opts, index=_idx(yn_opts, "pe", "no"))
                with h6:
                    f_ane = st.selectbox("Anemia", options=yn_opts, index=_idx(yn_opts, "ane", "no"))

                submit_manual = st.form_submit_button(
                    "🩺 Calculate CKD Risk & Generate Doctor + Patient Report",
                    type="primary", use_container_width=True,
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
                run_screening_dialog("manual", manual_payload, ml_model, groq_api_key, llm_model_name, str(time.time()))


def page_reports():
    st.markdown('<h1 class="page-title">Saved Patient Reports</h1><p class="page-sub">Search, filter, export and re-open any past screening.</p>', unsafe_allow_html=True)
    records = load_all_records()

    if not records:
        st.markdown(
            flat_html("""
            <div class="card" style="text-align:center; padding:2.5rem;">
                <h3 class="card-title">No Saved Patient Reports Yet</h3>
                <p class="card-sub">Run a screening to automatically save records to <code>patient_reports.json</code>.</p>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("🩺 Start a New Screening", type="primary", key="rep_start"):
            go(NAV_SCREEN)
        return

    total = len(records)
    ckd = sum(1 for r in records if r.get("prediction", {}).get("value") == 1)
    avg_conf = sum(r.get("prediction", {}).get("confidence", 0) for r in records) / total
    st.markdown(
        flat_html(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-ico blue">🧪</div><div><div class="stat-lbl">Total Screenings</div><div class="stat-val">{total}</div></div></div>
            <div class="stat-card"><div class="stat-ico red">⚠️</div><div><div class="stat-lbl">CKD Detected</div><div class="stat-val red">{ckd}</div></div></div>
            <div class="stat-card"><div class="stat-ico green">✅</div><div><div class="stat-lbl">Healthy Profiles</div><div class="stat-val green">{total - ckd}</div></div></div>
            <div class="stat-card"><div class="stat-ico teal">🎯</div><div><div class="stat-lbl">Avg Confidence</div><div class="stat-val blue">{avg_conf:.1f}%</div></div></div>
        </div>
        """),
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
        st.download_button("⬇️ Export Table (CSV)", data=registry_to_df(filtered).to_csv(index=False),
                           file_name="patient_registry.csv", mime="text/csv",
                           use_container_width=True, disabled=not filtered)
    with fc4:
        st.write("")
        st.write("")
        st.download_button("⬇️ Export Full Registry (JSON)", data=json.dumps(records, indent=2),
                           file_name="patient_reports_export.json", mime="application/json",
                           use_container_width=True)

    if not filtered:
        st.info("No saved reports match your search or filter.")
        return

    st.markdown(flat_html(build_registry_table(filtered)), unsafe_allow_html=True)
    st.caption(f"Showing {len(filtered)} of {total} reports • Red cells = abnormal values • Scroll sideways to see all 20 patient inputs")

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    label_map = {}
    for r in filtered:
        nm = str(r.get("user_data", {}).get("name") or "Anonymous Patient")
        pr = r.get("prediction", {})
        label_map[r.get("id")] = f"#{r.get('id')} — {nm} • {pr.get('label', '')} ({float(pr.get('confidence', 0)):.1f}%)"
    picked_id = st.selectbox("🔎 Open full Doctor & Patient summary for a report",
                             options=list(label_map.keys()), format_func=lambda i: label_map[i], key="history_pick")
    picked = next((r for r in filtered if r.get("id") == picked_id), None)
    if picked is not None:
        render_diagnostic_report(picked, key_prefix="hist")



# ============================================================
# 10A. COHORT-WIDE ANALYSIS — ALWAYS READS THE SAVED REGISTRY
# ============================================================
NUMERIC_ANALYSIS_FEATURES = {
    "age", "bp", "al", "su", "bgr", "bu", "sc", "hemo", "pcv", "rc"
}


def _analysis_value(feature: str, value):
    """Return a usable saved value without filling missing fields with defaults."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a", "na", "unknown"):
        return None
    if feature in NUMERIC_ANALYSIS_FEATURES:
        try:
            number = float(str(value).replace(",", "").strip())
            return number if pd.notna(number) else None
        except (TypeError, ValueError):
            return None
    return str(value).strip().lower()


def build_analysis_dataframe(records: list) -> pd.DataFrame:
    """Build one cohort row per saved screening, preserving genuinely missing values."""
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        user_data = record.get("user_data") if isinstance(record.get("user_data"), dict) else {}
        prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}

        raw_prediction = prediction.get("value")
        try:
            ckd_value = int(float(raw_prediction)) if raw_prediction is not None else None
        except (TypeError, ValueError):
            label_text = str(prediction.get("label", "")).strip().lower()
            ckd_value = 1 if label_text == "ckd" else (0 if label_text == "healthy" else None)

        try:
            confidence = float(prediction.get("confidence"))
        except (TypeError, ValueError):
            confidence = None

        row = {
            "Report ID": record.get("id"),
            "Patient": str(user_data.get("name") or "Anonymous Patient"),
            "Timestamp": pd.to_datetime(record.get("timestamp"), errors="coerce"),
            "Prediction": prediction.get("label") or ("CKD" if ckd_value == 1 else "Healthy" if ckd_value == 0 else "Unknown"),
            "CKD Target": ckd_value,
            "Confidence (%)": confidence,
        }
        for feature in FEATURES:
            row[feature] = _analysis_value(feature, user_data.get(feature))
            row[f"{feature}__abnormal"] = (
                row[feature] is not None and is_abnormal_biomarker(feature, row[feature])
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _analysis_corr_series(frame: pd.DataFrame, feature: str) -> pd.Series:
    values = frame[feature]
    if feature in CATEGORICAL_MAP:
        normalized = values.map(
            lambda x: str(x).strip().lower() if x is not None and not pd.isna(x) else None
        )
        return normalized.map(CATEGORICAL_MAP[feature])
    return pd.to_numeric(values, errors="coerce")


def build_biomarker_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize all 20 markers using observed values only."""
    total = len(frame)
    rows = []
    for feature in FEATURES:
        observed = frame[feature].dropna()
        available = len(observed)
        flagged = int(frame.loc[frame[feature].notna(), f"{feature}__abnormal"].sum())
        row = {
            "Biomarker": FEATURE_META[feature][0],
            "Unit": FEATURE_META[feature][1],
            "Available": available,
            "Missing": total - available,
            "Flagged": flagged,
            "Flagged %": round((flagged / available) * 100, 1) if available else None,
        }
        if feature in NUMERIC_ANALYSIS_FEATURES and available:
            numeric = pd.to_numeric(observed, errors="coerce").dropna()
            row.update({
                "Mean": round(float(numeric.mean()), 2) if len(numeric) else None,
                "Median": round(float(numeric.median()), 2) if len(numeric) else None,
                "Min": round(float(numeric.min()), 2) if len(numeric) else None,
                "Max": round(float(numeric.max()), 2) if len(numeric) else None,
                "Most common": "—",
            })
        else:
            modes = observed.astype(str).mode()
            row.update({
                "Mean": None, "Median": None, "Min": None, "Max": None,
                "Most common": modes.iloc[0] if not modes.empty else "—",
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _render_analysis_page():
    st.markdown(
        '<h1 class="page-title">Saved Data Analysis</h1>'
        '<p class="page-sub">Cohort-wide statistical intelligence, predictive dynamics, biomarker distributions, and correlation epidemiology.</p>',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.caption("Source: patient_reports.json • New screenings are aggregated automatically.")
    with top_right:
        if st.button("↻ Refresh Analysis", type="primary", use_container_width=True, key="analysis_refresh"):
            st.toast("Analysis refreshed from the saved registry.", icon="✅")

    # Deliberately not cached: every render reads the latest saved JSON registry.
    records = load_all_records()
    if not records:
        st.info("No saved screenings yet. Once a patient report is saved, its data will appear here automatically.")
        st.caption("This page checks for new saved reports every 30 seconds when the installed Streamlit version supports live fragments.")
        return

    all_frame = build_analysis_dataframe(records)
    if all_frame.empty:
        st.warning("The registry file contains no readable patient records.")
        return

    total = len(all_frame)
    known_predictions = all_frame["CKD Target"].dropna()
    ckd_count = int((known_predictions == 1).sum())
    healthy_count = int((known_predictions == 0).sum())
    unknown_count = total - ckd_count - healthy_count
    prediction_total = ckd_count + healthy_count
    ckd_rate = (ckd_count / prediction_total * 100) if prediction_total else 0.0
    healthy_rate = (healthy_count / prediction_total * 100) if prediction_total else 0.0
    confidence = pd.to_numeric(all_frame["Confidence (%)"], errors="coerce").dropna()
    avg_confidence = float(confidence.mean()) if not confidence.empty else 0.0
    completeness = sum(
        int(all_frame[f].notna().sum()) for f in FEATURES
    ) / (total * len(FEATURES)) * 100 if total else 0.0
    latest_timestamp = all_frame["Timestamp"].max()

    # 1. Executive Telemetry Cards (V2 Modern Design)
    st.markdown(flat_html(f"""
    <div class="ds-kpi-grid">
        <div class="ds-kpi-card blue">
            <div class="ds-kpi-header">
                <span class="ds-kpi-title">Saved Screenings</span>
                <span class="ds-kpi-icon">👥</span>
            </div>
            <div class="ds-kpi-val">{total}</div>
            <div class="ds-kpi-sub">Total patient registry records</div>
        </div>
        <div class="ds-kpi-card red">
            <div class="ds-kpi-header">
                <span class="ds-kpi-title">CKD Detections</span>
                <span class="ds-kpi-icon">⚠️</span>
            </div>
            <div class="ds-kpi-val red">{ckd_count}</div>
            <div class="ds-kpi-sub">{ckd_rate:.1f}% diagnosed with renal disease</div>
        </div>
        <div class="ds-kpi-card green">
            <div class="ds-kpi-header">
                <span class="ds-kpi-title">Healthy Cohort</span>
                <span class="ds-kpi-icon">🟢</span>
            </div>
            <div class="ds-kpi-val green">{healthy_count}</div>
            <div class="ds-kpi-sub">{healthy_rate:.1f}% baseline healthy profiles</div>
        </div>
        <div class="ds-kpi-card teal">
            <div class="ds-kpi-header">
                <span class="ds-kpi-title">Mean Confidence</span>
                <span class="ds-kpi-icon">🎯</span>
            </div>
            <div class="ds-kpi-val">{avg_confidence:.1f}%</div>
            <div class="ds-kpi-sub">Across {len(confidence)} scored screenings</div>
        </div>
        <div class="ds-kpi-card purple">
            <div class="ds-kpi-header">
                <span class="ds-kpi-title">Data Completeness</span>
                <span class="ds-kpi-icon">📋</span>
            </div>
            <div class="ds-kpi-val">{completeness:.1f}%</div>
            <div class="ds-kpi-sub">Across all 20 clinical input fields</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    if unknown_count:
        st.markdown(
            f'<div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:12px; padding:0.6rem 1rem; margin-bottom:1.1rem; color:#92400E; font-size:0.84rem; display:flex; align-items:center; gap:0.5rem;">'
            f'<span>ℹ️</span> <span><strong>{unknown_count} record</strong> has incomplete prediction telemetry and is excluded from diagnosis percentage calculations.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 2. Scope Filter & Latest Time
    scope_col, info_col = st.columns([1.2, 1.8])
    with scope_col:
        scope = st.selectbox(
            "Analysis Scope",
            [f"All saved records ({total})", f"CKD predictions only ({ckd_count})", f"Healthy predictions only ({healthy_count})"],
            key="analysis_scope",
        )
    with info_col:
        if pd.notna(latest_timestamp):
            st.caption(f"🕒 Latest saved screening: {latest_timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("🕒 Saved timestamps are unavailable for these records.")

    frame = all_frame.copy()
    if scope.startswith("CKD"):
        frame = frame[frame["CKD Target"] == 1].copy()
    elif scope.startswith("Healthy"):
        frame = frame[frame["CKD Target"] == 0].copy()

    if frame.empty:
        st.info("There are no records in the selected analysis scope.")
        return

    scope_known = frame["CKD Target"].dropna()
    scope_ckd = int((scope_known == 1).sum())
    scope_healthy = int((scope_known == 0).sum())
    scope_total_known = scope_ckd + scope_healthy
    scope_ckd_rate = (scope_ckd / scope_total_known * 100) if scope_total_known else 0.0
    scope_confidence = pd.to_numeric(frame["Confidence (%)"], errors="coerce").dropna()

    # 3. Data Science Insights Box
    biomarker_table = build_biomarker_analysis(frame)
    frequent_flags = biomarker_table[
        (biomarker_table["Available"] > 0) & (biomarker_table["Flagged"] > 0)
    ].sort_values(["Flagged %", "Flagged"], ascending=False)
    if not frequent_flags.empty:
        leading = frequent_flags.head(3)
        flag_summary = ", ".join(
            f"{r['Biomarker']} ({r['Flagged %']:.1f}%)"
            for _, r in leading.iterrows()
        )
    else:
        flag_summary = "All recorded biomarkers within normal baseline limits."

    age_values = pd.to_numeric(frame["age"], errors="coerce").dropna()
    age_median = f"{age_values.median():.1f} yrs" if not age_values.empty else "N/A"

    comorbidity_features = ["htn", "dm", "cad", "pe", "ane"]
    comorbidity_labels = {"htn": "Hypertension", "dm": "Diabetes", "cad": "Coronary artery disease", "pe": "Pedal edema", "ane": "Anemia"}
    comorbidity_counts = []
    for feature in comorbidity_features:
        vals = frame[feature].dropna()
        if len(vals):
            positives = int((vals.astype(str).str.lower() == "yes").sum())
            comorbidity_counts.append((comorbidity_labels[feature], positives, len(vals)))
    comorb_text = "N/A"
    if comorbidity_counts:
        most_common = sorted(comorbidity_counts, key=lambda item: item[1], reverse=True)[:2]
        comorb_text = "; ".join(f"{name} ({count}/{available})" for name, count, available in most_common)

    st.markdown(flat_html(f"""
    <div class="ds-findings-box">
        <div class="ds-findings-title">🔬 Key Cohort Data Science & Clinical Insights</div>
        <div class="ds-findings-grid">
            <div class="ds-finding-item">
                <span class="ds-finding-icon">🧬</span>
                <div><strong>Primary Biomarker Deviations:</strong> Most frequently elevated markers: {flag_summary}.</div>
            </div>
            <div class="ds-finding-item">
                <span class="ds-finding-icon">🫀</span>
                <div><strong>Comorbidity Prevalence:</strong> High co-occurrence marked yes most often: {comorb_text}.</div>
            </div>
            <div class="ds-finding-item">
                <span class="ds-finding-icon">🎯</span>
                <div><strong>Model Confidence Dynamic:</strong> Average certainty of <strong>{scope_confidence.mean():.1f}%</strong> across scored records in this scope.</div>
            </div>
            <div class="ds-finding-item">
                <span class="ds-finding-icon">👤</span>
                <div><strong>Cohort Demographics:</strong> Patient median age is <strong>{age_median}</strong> (recorded sample: {len(age_values)}).</div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # 4. Multi-Tab Comprehensive Analytics Suite
    tab_dynamics, tab_comparator, tab_correlation, tab_biomarkers, tab_data = st.tabs([
        "📊 Cohort & Predictive Dynamics",
        "⚖️ CKD vs. Healthy Comparison Lab",
        "🔬 Correlation & Scatter Lab",
        "🧪 20 Biomarkers Clinical Audit",
        "🗄️ Full Patient Cohort Explorer (All Data)"
    ])

    # ---------------- TAB 1: COHORT & PREDICTIVE DYNAMICS ----------------
    with tab_dynamics:
        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            st.markdown("#### Diagnostic Distribution (Cohort Ratio)")
            pred_counts = frame["Prediction"].value_counts().reset_index()
            pred_counts.columns = ["Diagnosis", "Count"]
            donut = alt.Chart(pred_counts).mark_arc(innerRadius=65, stroke="#fff", strokeWidth=2).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Diagnosis", type="nominal",
                                scale=alt.Scale(domain=["CKD", "Healthy", "Unknown"],
                                               range=["#E11D48", "#059669", "#94A3B8"]),
                                legend=alt.Legend(title="Diagnosis", orient="bottom")),
                tooltip=["Diagnosis:N", "Count:Q"]
            ).properties(height=260)
            st.altair_chart(donut, use_container_width=True)

        with chart_col2:
            st.markdown("#### Model Confidence Score Spread")
            conf_sub = frame[frame["Confidence (%)"].notna()].copy()
            if not conf_sub.empty:
                conf_hist = alt.Chart(conf_sub).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Confidence (%):Q", bin=alt.Bin(maxbins=12), title="Model Confidence (%)"),
                    y=alt.Y("count():Q", title="Patients"),
                    color=alt.Color("Prediction:N", scale=alt.Scale(domain=["CKD", "Healthy"], range=["#E11D48", "#059669"]), legend=alt.Legend(orient="bottom")),
                    tooltip=["count():Q", "Prediction:N"]
                ).properties(height=260)
                st.altair_chart(conf_hist, use_container_width=True)
            else:
                st.caption("No confidence scores available in this scope.")

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        age_col1, age_col2 = st.columns([1.1, 0.9], gap="large")
        with age_col1:
            st.markdown("#### Age Group vs. CKD Incidence")
            age_df = frame[frame["age"].notna()].copy()
            if not age_df.empty:
                age_df["age_num"] = pd.to_numeric(age_df["age"], errors="coerce")
                age_df["Age Bracket"] = pd.cut(age_df["age_num"], bins=[0, 39, 59, 120], labels=["Under 40", "40–59 yrs", "60+ yrs"])
                age_grouped = age_df.groupby(["Age Bracket", "Prediction"], observed=False).size().reset_index(name="Count")

                age_chart = alt.Chart(age_grouped).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Age Bracket:N", title="Patient Age Group"),
                    y=alt.Y("Count:Q", title="Patient Count"),
                    color=alt.Color("Prediction:N", scale=alt.Scale(domain=["CKD", "Healthy"], range=["#E11D48", "#059669"]), legend=alt.Legend(orient="bottom")),
                    xOffset="Prediction:N",
                    tooltip=["Age Bracket:N", "Prediction:N", "Count:Q"]
                ).properties(height=260)
                st.altair_chart(age_chart, use_container_width=True)
            else:
                st.caption("No age values recorded.")

        with age_col2:
            st.markdown("#### Screening Volume Over Time")
            trend = frame.dropna(subset=["Timestamp"]).copy()
            if not trend.empty:
                trend["Date"] = trend["Timestamp"].dt.date
                trend["Prediction group"] = trend["Prediction"].where(trend["Prediction"].isin(["CKD", "Healthy"]), "Unknown")
                daily = trend.groupby(["Date", "Prediction group"]).size().unstack(fill_value=0)
                for group in ["CKD", "Healthy"]:
                    if group not in daily.columns:
                        daily[group] = 0
                st.line_chart(daily[["CKD", "Healthy"]], height=260, use_container_width=True)
            else:
                st.caption("Timestamps needed to show screening timeline.")

    # ---------------- TAB 2: CKD VS. HEALTHY COMPARISON LAB ----------------
    with tab_comparator:
        st.markdown("#### Clinical Biomarker Comparator (CKD vs. Healthy Distributions)")
        st.caption("Select any numerical biomarker to inspect how its values diverge between diagnosed CKD patients and baseline healthy patients.")

        comp_biomarkers = [
            ("sc", "Serum Creatinine"),
            ("bu", "Blood Urea"),
            ("hemo", "Hemoglobin"),
            ("bgr", "Random Blood Glucose"),
            ("al", "Urine Albumin"),
            ("bp", "Blood Pressure"),
            ("pcv", "Packed Cell Volume"),
            ("rc", "RBC Count"),
            ("age", "Patient Age"),
        ]
        comp_keys = [k for k, _ in comp_biomarkers]
        comp_labels = [f"{lbl} ({k})" for k, lbl in comp_biomarkers]

        selected_idx = st.selectbox(
            "Select Biomarker to Compare",
            options=range(len(comp_biomarkers)),
            format_func=lambda i: comp_labels[i],
            key="comp_biomarker_select"
        )
        sel_key = comp_keys[selected_idx]
        sel_name, sel_unit, sel_ref = FEATURE_META[sel_key]

        comp_df = all_frame[all_frame["Prediction"].isin(["CKD", "Healthy"])].copy()
        comp_df["val_num"] = pd.to_numeric(comp_df[sel_key], errors="coerce")
        clean_comp = comp_df.dropna(subset=["val_num"])

        box_col, stats_col = st.columns([1.2, 0.8], gap="large")
        with box_col:
            if not clean_comp.empty:
                box_chart = alt.Chart(clean_comp).mark_boxplot(extent="min-max", size=50).encode(
                    x=alt.X("Prediction:N", title="Cohort Group", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("val_num:Q", title=f"{sel_name} ({sel_unit})"),
                    color=alt.Color("Prediction:N", scale=alt.Scale(domain=["CKD", "Healthy"], range=["#E11D48", "#059669"]), legend=None),
                    tooltip=["Prediction:N", alt.Tooltip("min(val_num):Q", format=".2f", title="Min"),
                             alt.Tooltip("median(val_num):Q", format=".2f", title="Median"),
                             alt.Tooltip("max(val_num):Q", format=".2f", title="Max")]
                ).properties(title=f"{sel_name}: Distribution by Clinical Diagnosis", height=300)
                st.altair_chart(box_chart, use_container_width=True)
            else:
                st.caption(f"No numeric values available for {sel_name}.")

        with stats_col:
            ckd_vals = clean_comp.loc[clean_comp["Prediction"] == "CKD", "val_num"]
            healthy_vals = clean_comp.loc[clean_comp["Prediction"] == "Healthy", "val_num"]
            ckd_mean = ckd_vals.mean() if not ckd_vals.empty else 0.0
            healthy_mean = healthy_vals.mean() if not healthy_vals.empty else 0.0

            diff_pct = ((ckd_mean - healthy_mean) / healthy_mean * 100) if healthy_mean != 0 else 0.0
            diff_icon = "🔺" if diff_pct > 0 else "🔻"
            diff_color = "#BE123C" if diff_pct > 0 else "#059669"

            st.markdown(flat_html(f"""
            <div style="background:#FFFFFF; border:1.5px solid #D9E8F2; border-radius:18px; padding:1.1rem 1.25rem; box-shadow:0 8px 18px -8px rgba(2,132,199,0.12);">
                <div style="font-size:0.74rem; font-weight:800; color:#0369A1; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.7rem;">📊 Comparative Telemetry Summary</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; font-size:0.86rem;">
                    <span style="color:#64748B;">CKD Group Mean:</span>
                    <strong style="color:#BE123C;">{ckd_mean:.2f} {sel_unit}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; font-size:0.86rem;">
                    <span style="color:#64748B;">Healthy Group Mean:</span>
                    <strong style="color:#059669;">{healthy_mean:.2f} {sel_unit}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.75rem; font-size:0.86rem; padding-top:0.4rem; border-top:1px dashed #E2E8F0;">
                    <span style="color:#64748B;">Cohort Divergence:</span>
                    <strong style="color:{diff_color};">{diff_icon} {abs(diff_pct):.1f}% relative difference</strong>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:0.5rem 0.8rem; font-size:0.76rem; color:#475569;">
                    <span>Standard Reference: <strong>{sel_ref}</strong></span>
                </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Comorbidity & Symptom Prevalence: CKD vs. Healthy")
        comorbidities = ["htn", "dm", "cad", "pe", "ane"]
        comorb_names = {"htn": "Hypertension", "dm": "Diabetes", "cad": "Coronary Artery Dis.", "pe": "Pedal Edema", "ane": "Anemia"}
        comorb_rows = []
        for c in comorbidities:
            for target, cl_label in [(1, "CKD"), (0, "Healthy")]:
                sub = all_frame[all_frame["CKD Target"] == target]
                if not sub.empty:
                    vals = sub[c].dropna().astype(str).str.lower()
                    yes_pct = (vals == "yes").sum() / len(vals) * 100 if len(vals) else 0.0
                    comorb_rows.append({"Comorbidity": comorb_names[c], "Diagnosis": cl_label, "Prevalence %": round(yes_pct, 1)})

        if comorb_rows:
            comorb_df = pd.DataFrame(comorb_rows)
            comorb_chart = alt.Chart(comorb_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Comorbidity:N", title=""),
                y=alt.Y("Prevalence %:Q", title="Cohort Prevalence (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Diagnosis:N", scale=alt.Scale(domain=["CKD", "Healthy"], range=["#E11D48", "#059669"]), legend=alt.Legend(orient="bottom")),
                xOffset="Diagnosis:N",
                tooltip=["Comorbidity:N", "Diagnosis:N", "Prevalence %:Q"]
            ).properties(title="Comorbidity Co-Occurrence: CKD vs. Healthy", height=270)
            st.altair_chart(comorb_chart, use_container_width=True)

    # ---------------- TAB 3: CORRELATION & SCATTER LAB ----------------
    with tab_correlation:
        st.markdown("#### Interactive Bi-Variate Correlation & Scatter Lab")
        st.caption("Select any two clinical biomarkers to analyze their cross-correlation, clinical clustering, and linear regression trajectory.")

        sc_c1, sc_c2 = st.columns(2)
        with sc_c1:
            x_feature = st.selectbox("X-Axis Biomarker", options=comp_keys, index=0, key="corr_x_feat", format_func=lambda k: FEATURE_META[k][0])
        with sc_c2:
            y_feature = st.selectbox("Y-Axis Biomarker", options=comp_keys, index=1, key="corr_y_feat", format_func=lambda k: FEATURE_META[k][0])

        x_name, x_unit, _ = FEATURE_META[x_feature]
        y_name, y_unit, _ = FEATURE_META[y_feature]

        scatter_df = all_frame[all_frame["Prediction"].isin(["CKD", "Healthy"])].copy()
        scatter_df["x_val"] = pd.to_numeric(scatter_df[x_feature], errors="coerce")
        scatter_df["y_val"] = pd.to_numeric(scatter_df[y_feature], errors="coerce")
        clean_scatter = scatter_df.dropna(subset=["x_val", "y_val"])

        if not clean_scatter.empty:
            scatter_base = alt.Chart(clean_scatter).encode(
                x=alt.X("x_val:Q", title=f"{x_name} ({x_unit})"),
                y=alt.Y("y_val:Q", title=f"{y_name} ({y_unit})")
            )
            points = scatter_base.mark_circle(size=130, opacity=0.85, stroke="#334155", strokeWidth=1).encode(
                color=alt.Color("Prediction:N", scale=alt.Scale(domain=["CKD", "Healthy"], range=["#E11D48", "#059669"]), legend=alt.Legend(orient="bottom")),
                tooltip=["Patient:N", alt.Tooltip("x_val:Q", format=".2f", title=x_name), alt.Tooltip("y_val:Q", format=".2f", title=y_name), "Prediction:N"]
            )
            line = scatter_base.transform_regression("x_val", "y_val").mark_line(color="#0284C7", strokeDash=[4, 4], strokeWidth=2)
            scatter_combined = (points + line).properties(title=f"Bi-Variate Regression: {x_name} vs. {y_name}", height=320)
            st.altair_chart(scatter_combined, use_container_width=True)
        else:
            st.caption("Insufficient valid data points to render scatter regression.")

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        hm_col1, hm_col2 = st.columns([1.1, 0.9], gap="large")
        with hm_col1:
            st.markdown("#### Biomarker Spearman Correlation Matrix")
            num_cols = ["age", "bp", "bgr", "bu", "sc", "hemo", "pcv", "rc", "al"]
            corr_matrix = all_frame[num_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").reset_index()
            corr_melted = corr_matrix.melt(id_vars="index")
            corr_melted.columns = ["Biomarker 1", "Biomarker 2", "Spearman Correlation"]
            corr_melted["Spearman Correlation"] = corr_melted["Spearman Correlation"].round(2)

            heatmap = alt.Chart(corr_melted).mark_rect().encode(
                x=alt.X("Biomarker 1:N", title="", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Biomarker 2:N", title=""),
                color=alt.Color("Spearman Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), legend=alt.Legend(orient="bottom")),
                tooltip=["Biomarker 1", "Biomarker 2", "Spearman Correlation"]
            ).properties(height=320)

            heat_text = alt.Chart(corr_melted).mark_text(fontSize=9, fontWeight=600).encode(
                x="Biomarker 1:N",
                y="Biomarker 2:N",
                text=alt.Text("Spearman Correlation:Q", format=".2f"),
                color=alt.condition(
                    "abs(datum['Spearman Correlation']) > 0.5",
                    alt.value("#FFFFFF"),
                    alt.value("#0F172A")
                )
            )
            st.altair_chart(heatmap + heat_text, use_container_width=True)

        with hm_col2:
            st.markdown("#### Ensemble Model Feature Importance")
            importances = [
                ("Hemoglobin (hemo)", 24.7),
                ("Urine Albumin (al)", 21.2),
                ("Packed Cell Volume (pcv)", 17.7),
                ("RBC Count (rc)", 16.1),
                ("Serum Creatinine (sc)", 6.4),
                ("Blood Glucose (bgr)", 4.8),
                ("Blood Urea (bu)", 3.8),
                ("Blood Pressure (bp)", 2.9),
                ("Age", 2.4),
            ]
            fi_df = pd.DataFrame(importances, columns=["Feature", "Importance %"])
            fi_chart = alt.Chart(fi_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color="#0284C7").encode(
                y=alt.Y("Feature:N", sort="-x", title=""),
                x=alt.X("Importance %:Q", title="Model Split Weight (%)"),
                tooltip=["Feature:N", alt.Tooltip("Importance %:Q", format=".1f")]
            ).properties(height=320)
            st.altair_chart(fi_chart, use_container_width=True)

    # ---------------- TAB 4: 20 BIOMARKERS CLINICAL AUDIT ----------------
    with tab_biomarkers:
        st.markdown("#### Top 10 Flagged Abnormal Biomarkers in Cohort")
        top_flags = biomarker_table[biomarker_table["Available"] > 0].sort_values("Flagged %", ascending=False).head(10).copy()
        if not top_flags.empty:
            bar_flags = alt.Chart(top_flags).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                y=alt.Y("Biomarker:N", sort="-x", title=""),
                x=alt.X("Flagged %:Q", title="Cohort Flagged Abnormal (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Flagged %:Q", scale=alt.Scale(scheme="oranges", domain=[0, 100]), legend=None),
                tooltip=["Biomarker:N", alt.Tooltip("Flagged %:Q", format=".1f"), "Flagged:Q", "Available:Q"]
            ).properties(height=280)
            st.altair_chart(bar_flags, use_container_width=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Full 20 Biomarkers Statistical Audit Table")
        st.caption("Observed values across registry records. Missing values are excluded from statistical means and medians.")
        display_columns = ["Biomarker", "Unit", "Available", "Missing", "Flagged", "Flagged %", "Mean", "Median", "Min", "Max", "Most common"]
        st.dataframe(
            biomarker_table[display_columns],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Flagged %": st.column_config.NumberColumn(format="%.1f%%"),
                "Mean": st.column_config.NumberColumn(format="%.2f"),
                "Median": st.column_config.NumberColumn(format="%.2f"),
                "Min": st.column_config.NumberColumn(format="%.2f"),
                "Max": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # ---------------- TAB 5: FULL PATIENT COHORT EXPLORER (ALL DATA) ----------------
    with tab_data:
        st.markdown("#### Complete Patient Cohort Telemetry (All Records & 20 Biomarkers)")
        st.caption("Search, filter, and inspect all patient records with full clinical inputs and ML diagnostics.")

        f_c1, f_c2 = st.columns([2, 1])
        with f_c1:
            search_query = st.text_input("🔍 Search patient by name...", key="cohort_search_query", placeholder="Type a patient name...")
        with f_c2:
            diag_filter = st.selectbox("Filter Diagnosis", ["All Patients", "CKD Only", "Healthy Only"], key="cohort_diag_filter")

        table_rows = []
        for r in records:
            ud = r.get("user_data", {})
            pred_info = r.get("prediction") or {}
            row_dict = {
                "Report ID": r.get("id"),
                "Patient": str(ud.get("name") or "Anonymous"),
                "Timestamp": str(r.get("timestamp") or "—")[:16].replace("T", " "),
                "Diagnosis": pred_info.get("label", "Unknown"),
                "Confidence (%)": f"{float(pred_info.get('confidence', 0)):.1f}%" if pred_info.get("confidence") is not None else "—",
                "Age": ud.get("age", "—"),
                "BP": ud.get("bp", "—"),
                "Creatinine (sc)": ud.get("sc", "—"),
                "Urea (bu)": ud.get("bu", "—"),
                "Hemoglobin (hemo)": ud.get("hemo", "—"),
                "Glucose (bgr)": ud.get("bgr", "—"),
                "Albumin (al)": ud.get("al", "—"),
                "Sugar (su)": ud.get("su", "—"),
                "RBC": ud.get("rbc", "—"),
                "Pus Cells (pc)": ud.get("pc", "—"),
                "PC Clumps": ud.get("pcc", "—"),
                "Bacteria": ud.get("ba", "—"),
                "PCV": ud.get("pcv", "—"),
                "WBC (wc)": ud.get("wc", "—"),
                "RBC Count (rc)": ud.get("rc", "—"),
                "HTN": ud.get("htn", "—"),
                "Diabetes (dm)": ud.get("dm", "—"),
                "CAD": ud.get("cad", "—"),
                "Appetite": ud.get("appet", "—"),
                "Pedal Edema": ud.get("pe", "—"),
                "Anemia": ud.get("ane", "—"),
            }
            table_rows.append(row_dict)

        table_df = pd.DataFrame(table_rows)

        if search_query.strip():
            table_df = table_df[table_df["Patient"].str.contains(search_query.strip(), case=False, na=False)]
        if diag_filter == "CKD Only":
            table_df = table_df[table_df["Diagnosis"] == "CKD"]
        elif diag_filter == "Healthy Only":
            table_df = table_df[table_df["Diagnosis"] == "Healthy"]

        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            st.markdown(f"**Showing {len(table_df)} of {len(records)} cohort records**")
        with d_col2:
            csv_bytes = table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Cohort Dataset (CSV)",
                data=csv_bytes,
                file_name="nephrocare_patient_cohort.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

        st.dataframe(table_df, hide_index=True, use_container_width=True)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    st.warning(
        "Clinical & Data Science Notice: Cohort distributions and statistical associations presented here are derived from saved screening reports. "
        "These analyses are intended for clinical quality auditing, cohort epidemiology, and research intelligence, and do not replace personalized nephrology consultations."
    )
    st.caption("Live registry synchronization: new screenings saved from the New Screening pipeline will automatically reflect in this dataset upon refresh.")



# Use a Streamlit fragment for periodic registry refresh on supported versions.
# The fallback remains fully functional and refreshes on normal app reruns.
if hasattr(st, "fragment"):
    page_analysis = st.fragment(run_every="30s")(_render_analysis_page)
else:
    page_analysis = _render_analysis_page

def page_about():
    st.markdown('<h1 class="page-title">About NephroCare AI</h1><p class="page-sub">How the system works.</p>', unsafe_allow_html=True)
    st.markdown(
        flat_html("""
        <div class="feat-row">
            <div class="feat-card"><div class="fi">1️⃣</div><b>Extract</b><span>An LLM (Groq) reads the clinical note and structures 20 kidney biomarkers.</span></div>
            <div class="feat-card"><div class="fi">2️⃣</div><b>Predict</b><span>An ensemble of decision trees classifies the profile as CKD or healthy.</span></div>
            <div class="feat-card"><div class="fi">3️⃣</div><b>Explain</b><span>The LLM writes a doctor summary and a simple patient guide.</span></div>
            <div class="feat-card"><div class="fi">4️⃣</div><b>Save</b><span>Every result is stored in the registry and can be exported as JSON, CSV or Markdown.</span></div>
        </div>
        <div class="card" style="margin-top:1.2rem;">
            <span class="pill-tag">Disclaimer</span>
            <p class="readable-paragraph">NephroCare AI is a screening and decision-support tool. It does not replace a doctor's diagnosis.
            Always confirm results with a qualified nephrologist or physician.</p>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ============================================================
# 11. ROUTER
# ============================================================
page = st.session_state.nav
if page == NAV_HOME:
    page_home()
elif page == NAV_DASH:
    page_dashboard()
elif page == NAV_SCREEN:
    page_screening()
elif page == NAV_REPORTS:
    page_reports()
elif page == NAV_ANALYSIS:
    page_analysis()
else:
    page_about()
