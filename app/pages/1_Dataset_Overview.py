import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_dataset, preprocess, FEATURE_NAMES

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")

# CSS 
st.markdown("""
<style>
    .page-title {
        font-size: 2rem; font-weight: 800;
        color: #1565C0; margin-bottom: 0.2rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700;
        color: #1565C0; border-left: 4px solid #42A5F5;
        padding-left: 0.7rem; margin: 1.5rem 0 0.8rem 0;
    }
    .info-card {
        background: #F8FAFF; border: 1px solid #E3F2FD;
        border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 0.8rem;
        color: #1a1a1a;
    }
    .metric-box {
        background: white; border-radius: 12px;
        border: 1px solid #E3F2FD; padding: 1rem;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 1.8rem; font-weight: 800; color: #1565C0; }
    .metric-lbl { font-size: 0.8rem; color: #78909C; font-weight: 600; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0D47A1,#1565C0); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("##  Diabetes")
    st.markdown("---")
    st.markdown("""
    <div style='color:white;'>
    <b>Menu:</b><br>
    Home<br>
    <b>Dataset Overview ←</b><br>
    Prediction<br>
    Visualization<br>
    About
    </div>
    """, unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw  = load_dataset()
df_clean = preprocess(df_raw)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title"> Dataset Overview</div>', unsafe_allow_html=True)
st.markdown("Eksplorasi lengkap dataset **Pima Indians Diabetes Database** sebelum dan sesudah preprocessing.")

# ── Key metrics ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (str(df_raw.shape[0]), "Total Data"),
    (str(df_raw.shape[1] - 1), "Jumlah Fitur"),
    (str(df_raw["Outcome"].sum()), "Kasus Diabetes"),
    (str((df_raw["Outcome"] == 0).sum()), "Non-Diabetes"),
    (f"{df_raw['Outcome'].mean()*100:.1f}%", "Prevalensi Diabetes"),
]
for col, (val, lbl) in zip([m1, m2, m3, m4, m5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Dataset info ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Informasi Dataset</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 1])
with col_left:
    st.markdown("""
    <div class="info-card">
    <b>Nama Dataset:</b> Pima Indians Diabetes Database<br>
    <b>Sumber:</b> UCI Machine Learning Repository / Kaggle<br>
    <b>Jumlah Baris:</b> 768 data pasien<br>
    <b>Jumlah Kolom:</b> 9 (8 fitur + 1 target)<br>
    <b>Target:</b> Outcome — 0 (Tidak Diabetes) / 1 (Diabetes)<br>
    <b>Populasi:</b> Wanita keturunan Indian Pima, usia ≥ 21 tahun
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="info-card">
    <b>Fitur Medis:</b><br>
    • Pregnancies — Jumlah kehamilan<br>
    • Glucose — Kadar glukosa darah<br>
    • BloodPressure — Tekanan darah diastolik<br>
    • SkinThickness — Ketebalan lipatan kulit<br>
    • Insulin — Kadar insulin 2 jam<br>
    • BMI — Indeks massa tubuh<br>
    • DiabetesPedigreeFunction — Riwayat genetik diabetes<br>
    • Age — Usia (tahun)
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([" Preview Data", " Statistik Deskriptif", " Missing Values", " Distribusi Fitur"])

# Tab 1 — Preview
with tab1:
    sub1, sub2 = st.columns(2)
    with sub1:
        st.markdown("**Data Mentah (5 baris pertama)**")
        st.dataframe(df_raw.head(), use_container_width=True)
    with sub2:
        st.markdown("**Data Setelah Preprocessing (5 baris pertama)**")
        st.dataframe(df_clean.head(), use_container_width=True)

    st.markdown("**Tipe Data & Info Kolom**")
    info_df = pd.DataFrame({
        "Kolom"  : df_raw.columns,
        "Tipe"   : df_raw.dtypes.astype(str).values,
        "Non-Null": df_raw.notnull().sum().values,
        "Null"   : df_raw.isnull().sum().values,
        "Unique" : df_raw.nunique().values,
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

# Tab 2 — Stats
with tab2:
    st.markdown("**Statistik Deskriptif — Data Setelah Preprocessing**")
    desc = df_clean.describe().T.round(3)
    desc.index.name = "Fitur"
    st.dataframe(desc, use_container_width=True)

    st.markdown("---")
    st.markdown("**Perbandingan Rata-rata: Diabetes vs Tidak Diabetes**")
    compare = df_clean.groupby("Outcome")[FEATURE_NAMES].mean().round(2).T
    compare.columns = ["Tidak Diabetes (0)", "Diabetes (1)"]
    compare["Selisih"] = (compare["Diabetes (1)"] - compare["Tidak Diabetes (0)"]).round(2)
    compare["Selisih %"] = ((compare["Selisih"] / compare["Tidak Diabetes (0)"]) * 100).round(1).astype(str) + "%"
    st.dataframe(compare, use_container_width=True)

# Tab 3 — Missing Values
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Missing Values — Data Mentah**")
        mv_raw = df_raw.isnull().sum().reset_index()
        mv_raw.columns = ["Kolom", "Null Count"]
        mv_raw["Status"] = mv_raw["Null Count"].apply(lambda x: "✅ OK" if x == 0 else "⚠️ Ada Null")
        st.dataframe(mv_raw, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Nilai Nol pada Kolom Medis (sebelum impute)**")
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        zv = pd.DataFrame({
            "Kolom": zero_cols,
            "Jumlah Nol": [int((df_raw[c] == 0).sum()) for c in zero_cols],
            "% dari Total": [f"{(df_raw[c]==0).mean()*100:.1f}%" for c in zero_cols],
        })
        st.dataframe(zv, use_container_width=True, hide_index=True)

    st.info(" **Penanganan:** Nilai 0 pada kolom Glucose, BloodPressure, SkinThickness, Insulin, dan BMI merupakan missing values tersirat. Nilai tersebut diimputasi dengan rata-rata kolom masing-masing.")

# Tab 4 — Distribution
with tab4:
    st.markdown("**Distribusi Semua Fitur (Histogram)**")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    colors = ["#1565C0", "#1976D2", "#1E88E5", "#2196F3",
              "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"]
    all_cols = FEATURE_NAMES + ["Outcome"]
    for i, col in enumerate(all_cols):
        if col == "Outcome":
            counts = df_clean["Outcome"].value_counts()
            bars = axes[i].bar(["Tidak Diabetes (0)", "Diabetes (1)"],
                               counts.values, color=["#1565C0", "#E53935"], width=0.5)
            for bar, v in zip(bars, counts.values):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 3, str(v),
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
            axes[i].set_title("Outcome (Target)", fontsize=11, fontweight='bold')
        else:
            axes[i].hist(df_clean[col], bins=25, color=colors[i % len(colors)],
                         edgecolor='white', alpha=0.85)
            axes[i].axvline(df_clean[col].mean(), color='red', linestyle='--',
                            linewidth=1.2, alpha=0.7, label='Mean')
            axes[i].set_title(col, fontsize=11, fontweight='bold')
            axes[i].legend(fontsize=8)
        axes[i].set_ylabel("Frekuensi", fontsize=9)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].spines[['top', 'right']].set_visible(False)

    plt.suptitle("Distribusi Seluruh Fitur Dataset", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Heatmap Korelasi**")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
                ax=ax2, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9})
    ax2.set_title("Korelasi Antar Fitur", fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("**Boxplot Fitur berdasarkan Outcome**")
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
    axes3 = axes3.flatten()
    for i, feat in enumerate(FEATURE_NAMES):
        data_0 = df_clean[df_clean["Outcome"] == 0][feat]
        data_1 = df_clean[df_clean["Outcome"] == 1][feat]
        bp = axes3[i].boxplot([data_0, data_1], patch_artist=True,
                              labels=["Tidak\nDiabetes", "Diabetes"])
        bp["boxes"][0].set_facecolor("#90CAF9")
        bp["boxes"][1].set_facecolor("#EF9A9A")
        for box in bp["boxes"]:
            box.set_alpha(0.8)
        axes3[i].set_title(feat, fontsize=10, fontweight='bold')
        axes3[i].grid(axis='y', alpha=0.3)
        axes3[i].spines[['top', 'right']].set_visible(False)
    plt.suptitle("Distribusi Fitur: Diabetes vs Tidak Diabetes", fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)
