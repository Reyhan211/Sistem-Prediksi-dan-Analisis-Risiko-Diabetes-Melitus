import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",  
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary: #1565C0;
        --secondary: #42A5F5;
        --accent: #E53935;
        --success: #43A047;
        --bg-card: #F8FAFF;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1565C0, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #546E7A;
        margin-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #1565C0 0%, #1976D2 50%, #42A5F5 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(21, 101, 192, 0.3);
    }

    .hero-card h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white !important;
    }

    .hero-card p {
        font-size: 1.1rem;
        opacity: 0.92;
        line-height: 1.7;
        color: white !important;
    }

    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E3F2FD;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(21, 101, 192, 0.15);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.4rem;
    }

    .feature-desc {
        font-size: 0.9rem;
        color: #607D8B;
        line-height: 1.5;
    }

    .member-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        border-left: 4px solid #1565C0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .member-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1565C0;
    }

    .member-nim {
        font-size: 0.9rem;
        color: #78909C;
        font-family: monospace;
    }

    .member-role {
        display: inline-block;
        background: #E3F2FD;
        color: #1565C0;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    .stat-badge {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #90CAF9;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #1565C0;
        display: block;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #546E7A;
        font-weight: 600;
    }

    .method-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 4px;
    }

    .badge-rf { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
    .badge-km { background: #FFF3E0; color: #E65100; border: 1px solid #FFCC80; }
    .badge-if { background: #FCE4EC; color: #880E4F; border: 1px solid #F48FB1; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D47A1 0%, #1565C0 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }

    div[data-testid="stSidebarNav"] a span {
        color: white !important;
    }

    .divider {
        border: none;
        border-top: 1px solid #E3F2FD;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Diabetes")
    st.markdown("---")
    st.markdown("""
    **Navigasi:**
    - **Home** 
    - Dataset Overview
    - Prediction
    - Visualization
    - About
    """)
    st.markdown("---")
    st.markdown("""
    <div style='color: white; font-size: 0.8rem; opacity: 0.8;'>
    UAS Data Mining<br>
    Semester Genap 2026
    </div>
    """, unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1> Diabetes Prediction System</h1>
    <p>
        Sistem analisis prediksi diabetes berbasis Machine Learning yang mengintegrasikan
        tiga metode utama: <strong>Klasifikasi (Random Forest)</strong>, 
        <strong>Clustering (K-Means)</strong>, dan 
        <strong>Anomaly Detection (Isolation Forest)</strong> untuk mendukung
        pengambilan keputusan klinis berbasis data.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-badge">
        <span class="stat-number">768</span>
        <span class="stat-label">Total Data Pasien</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-badge">
        <span class="stat-number">8</span>
        <span class="stat-label">Fitur Medis</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-badge">
        <span class="stat-number">3</span>
        <span class="stat-label">Metode ML</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-badge">
        <span class="stat-number">~79%</span>
        <span class="stat-label">Akurasi Model</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Description ───────────────────────────────────────────────────────────────
st.markdown("### Deskripsi Proyek")
st.markdown("""
Proyek ini merupakan implementasi **Data Mining** menggunakan dataset **Pima Indians Diabetes Database**
dari Kaggle/UCI Machine Learning Repository. Dataset memuat data rekam medis wanita keturunan Indian 
Pima berusia minimal 21 tahun.

Tiga analisis utama dilakukan:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🟡</div>
        <div class="feature-title">Klasifikasi — Random Forest</div>
        <div class="feature-desc">
            Memprediksi apakah pasien menderita diabetes (positif/negatif)
            menggunakan ensemble 100 Decision Tree dengan hyperparameter tuning 
            via RandomizedSearchCV.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔵</div>
        <div class="feature-title">Clustering — K-Means</div>
        <div class="feature-desc">
            Mengelompokkan pasien ke dalam cluster berdasarkan kemiripan 
            profil kesehatan. K optimal dipilih menggunakan Elbow Method 
            dan Silhouette Score.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔴</div>
        <div class="feature-title">Anomaly Detection — Isolation Forest</div>
        <div class="feature-desc">
            Mendeteksi pasien dengan profil medis ekstrem atau tidak normal
            yang berpotensi menunjukkan kondisi kritis atau outlier data.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Team Members ──────────────────────────────────────────────────────────────
st.markdown("### Identitas Anggota Kelompok")

members = [
    {"name": "Ahnaf Reyhan Kenneth Alvaro", "nim": "24051214179"},
    {"name": "Elang Febriansyah",      "nim": "24051214202"},
]

cols = st.columns(2)
for i, m in enumerate(members):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="member-card">
            <div class="member-name">{m['name']}</div>
            <div class="member-nim">NIM: {m['nim']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("###  Teknologi yang Digunakan")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    | Library | Fungsi |
    |---------|--------|
    | `pandas` / `numpy` | Manipulasi data |
    | `scikit-learn` | Model ML |
    | `matplotlib` / `seaborn` | Visualisasi |
    | `streamlit` | Web App |
    | `pickle` | Simpan / Load model |
    """)

with col2:
    st.markdown("""
    | Algoritma | Kategori |
    |-----------|----------|
    | Random Forest | Classification |
    | K-Means | Clustering |
    | Isolation Forest | Anomaly Detection |
    | PCA | Dimensionality Reduction |
    | MinMaxScaler | Preprocessing |
    """)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#90A4AE; font-size:0.85rem; padding: 1rem 0;'>
     Diabetes Prediction System — UAS Data Mining 2026 &nbsp;|&nbsp; 
    Dibuat dengan menggunakan Python & Streamlit
</div>
""", unsafe_allow_html=True)
