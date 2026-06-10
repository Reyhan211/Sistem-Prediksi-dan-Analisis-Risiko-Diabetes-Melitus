import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_models, preprocess, load_dataset, FEATURE_NAMES

st.set_page_config(page_title="Prediction & Analysis", page_icon="🔮", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .page-title { font-size:2rem; font-weight:800; color:#1565C0; margin-bottom:.2rem; }
    .section-header {
        font-size:1.2rem; font-weight:700; color:#1565C0;
        border-left:4px solid #42A5F5; padding-left:.7rem; margin:1.5rem 0 .8rem 0;
    }
    .result-positive {
        background: linear-gradient(135deg,#FFEBEE,#FFCDD2);
        border:2px solid #EF9A9A; border-radius:16px;
        padding:1.5rem 2rem; text-align:center;
    }
    .result-negative {
        background: linear-gradient(135deg,#E8F5E9,#C8E6C9);
        border:2px solid #A5D6A7; border-radius:16px;
        padding:1.5rem 2rem; text-align:center;
    }
    .result-title { font-size:1.6rem; font-weight:800; margin-bottom:.4rem; }
    .result-sub   { font-size:.95rem; color:#546E7A; }
    .cluster-card {
        background: white; border-radius:14px; border:1px solid #E3F2FD;
        padding:1.2rem 1.5rem; box-shadow:0 2px 8px rgba(0,0,0,.05);
    }
    .anomaly-normal   { background:#E3F2FD; border-radius:12px; padding:.8rem 1.2rem; }
    .anomaly-detected { background:#FFEBEE; border-radius:12px; padding:.8rem 1.2rem; }
    .input-tip { font-size:.78rem; color:#90A4AE; }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0D47A1,#1565C0); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("##  Diabetes")
    st.markdown("---")
    st.markdown("""
    <div style='color:white;'>
    <b>Menu:</b><br> Home<br> Dataset Overview<br>
     <b>Prediction ←</b><br> Visualization<br> About
    </div>""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=" Melatih model, harap tunggu...")
def load_all():
    return get_models()

sc, clf, km, iso, X_all, df_clean, best_k = load_all()

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title"> Prediction & Analysis</div>', unsafe_allow_html=True)
st.markdown("Masukkan data medis pasien untuk mendapatkan prediksi **klasifikasi**, **cluster**, dan **deteksi anomali**.")

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Form Input Data Pasien</div>', unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    st.markdown("Isi semua field di bawah ini:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=2,
            help="Jumlah kehamilan")
        st.markdown('<p class="input-tip">Range normal: 0 – 17</p>', unsafe_allow_html=True)

        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=120,
            help="Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral")
        st.markdown('<p class="input-tip">Normal: 70–140 mg/dL</p>', unsafe_allow_html=True)

    with col2:
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)", min_value=0, max_value=150, value=70,
            help="Tekanan darah diastolik")
        st.markdown('<p class="input-tip">Normal: 60–90 mmHg</p>', unsafe_allow_html=True)

        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=100, value=23,
            help="Ketebalan lipatan kulit trisep")
        st.markdown('<p class="input-tip">Normal: 10–50 mm</p>', unsafe_allow_html=True)

    with col3:
        insulin = st.number_input(
            "Insulin (μU/mL)", min_value=0, max_value=900, value=80,
            help="Kadar insulin serum 2 jam")
        st.markdown('<p class="input-tip">Normal: 16–166 μU/mL</p>', unsafe_allow_html=True)

        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.0, max_value=80.0, value=32.0, step=0.1,
            help="Indeks massa tubuh = berat (kg) / tinggi² (m)")
        st.markdown('<p class="input-tip">Normal: 18.5–24.9</p>', unsafe_allow_html=True)

    with col4:
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.000, max_value=3.000,
            value=0.471, step=0.001, format="%.3f",
            help="Fungsi riwayat keluarga diabetes")
        st.markdown('<p class="input-tip">Range: 0.078 – 2.420</p>', unsafe_allow_html=True)

        age = st.number_input(
            "Age (tahun)", min_value=21, max_value=100, value=33,
            help="Usia pasien (minimal 21 tahun)")
        st.markdown('<p class="input-tip">Minimal 21 tahun</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(" Proses Prediksi", use_container_width=True, type="primary")

# ── Prediction Results ─────────────────────────────────────────────────────────
if submitted:
    # Build raw input array
    raw_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]], dtype=float)

    # Scale using fitted scaler (only features, not outcome)
    # scaler was fit on all 9 cols; we need to scale just the 8 features
    # We'll reconstruct: append dummy outcome=0, scale, then drop
    raw_with_outcome = np.hstack([raw_input, [[0]]])
    scaled_full = sc.transform(raw_with_outcome)
    scaled_features = scaled_full[:, :8]

    # ── Classification ─────────────────────────────────────────────────────────
    prediction  = clf.predict(scaled_features)[0]
    probability = clf.predict_proba(scaled_features)[0]

    # ── Clustering ─────────────────────────────────────────────────────────────
    cluster_label = int(km.predict(scaled_features)[0])

    # Cluster descriptions based on training data
    df_cl = df_clean.copy()
    X_all_clean = np.nan_to_num(X_all, nan=0.0)
    df_cl["Cluster"] = km.predict(X_all_clean)
    cluster_stats = df_cl.groupby("Cluster")["Outcome"].mean() * 100
    cluster_info = {
        c: f"{cluster_stats[c]:.1f}% diabetes" if c in cluster_stats else "N/A"
        for c in range(best_k)
    }
    cluster_colors = ["#1565C0", "#E65100", "#2E7D32"]
    cluster_names  = ["Cluster Risiko Rendah", "Cluster Risiko Tinggi", "Cluster Risiko Sedang"]

    # ── Anomaly ────────────────────────────────────────────────────────────────
    anomaly_pred  = iso.predict(scaled_features)[0]   # 1=normal, -1=anomali
    anomaly_score = iso.decision_function(scaled_features)[0]
    is_anomaly    = anomaly_pred == -1

    # ── Display Results ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header"> Hasil Prediksi</div>', unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns(3)

    # --- Classification result ---
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-title"> DIABETES TERDETEKSI</div>
                <div style="font-size:2.4rem; font-weight:900; color:#C62828;">
                    {probability[1]*100:.1f}%
                </div>
                <div class="result-sub">Probabilitas Positif Diabetes</div>
                <hr style="border-color:#FFCDD2; margin:.8rem 0;">
                <div style="font-size:.85rem; color:#B71C1C;">
                     Model memprediksi pasien berisiko diabetes.<br>
                    Disarankan pemeriksaan lebih lanjut.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-title"> TIDAK DIABETES</div>
                <div style="font-size:2.4rem; font-weight:900; color:#1B5E20;">
                    {probability[0]*100:.1f}%
                </div>
                <div class="result-sub">Probabilitas Negatif Diabetes</div>
                <hr style="border-color:#C8E6C9; margin:.8rem 0;">
                <div style="font-size:.85rem; color:#2E7D32;">
                     Model memprediksi pasien tidak menderita diabetes.<br>
                    Tetap jaga pola hidup sehat.
                </div>
            </div>""", unsafe_allow_html=True)
        st.caption("Metode: Random Forest (Tuned)")

    # --- Clustering result ---
    with res_col2:
        cidx   = cluster_label % len(cluster_names)
        ccolor = cluster_colors[cidx]
        cname  = cluster_names[cidx]
        cdesc  = cluster_info.get(cluster_label, "N/A")
        st.markdown(f"""
        <div class="cluster-card" style="border-left:5px solid {ccolor};">
            <div style="font-size:1rem; font-weight:700; color:{ccolor};">
                 Hasil Clustering
            </div>
            <div style="font-size:2rem; font-weight:900; color:{ccolor}; margin:.4rem 0;">
                Cluster {cluster_label}
            </div>
            <div style="font-size:1rem; font-weight:600; color:#37474F;">{cname}</div>
            <hr style="border-color:#E3F2FD; margin:.6rem 0;">
            <div style="font-size:.85rem; color:#607D8B;">
                 Prevalensi diabetes dalam cluster ini: <b>{cdesc}</b><br><br>
                Pasien dikelompokkan bersama pasien dengan profil kesehatan serupa.
            </div>
        </div>""", unsafe_allow_html=True)
        st.caption("Metode: K-Means (k=3)")

    # --- Anomaly result ---
    with res_col3:
        if is_anomaly:
            st.markdown(f"""
            <div class="anomaly-detected">
                <div style="font-size:1rem; font-weight:700; color:#C62828;">🔴 Anomaly Detection</div>
                <div style="font-size:1.8rem; font-weight:900; color:#C62828; margin:.4rem 0;">
                     ANOMALI
                </div>
                <div style="font-size:.9rem; font-weight:600;">Data Terdeteksi Tidak Normal</div>
                <hr style="border-color:#FFCDD2; margin:.6rem 0;">
                <div style="font-size:.82rem; color:#B71C1C;">
                    Anomaly score: <b>{anomaly_score:.4f}</b><br>
                    (Semakin negatif = semakin anomali)<br><br>
                    Data pasien ini memiliki nilai fitur yang sangat berbeda dari pola umum. 
                    Perlu verifikasi ulang data atau perhatian medis khusus.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="anomaly-normal">
                <div style="font-size:1rem; font-weight:700; color:#1565C0;"> Anomaly Detection</div>
                <div style="font-size:1.8rem; font-weight:900; color:#1565C0; margin:.4rem 0;">
                     NORMAL
                </div>
                <div style="font-size:.9rem; font-weight:600;">Data dalam Batas Normal</div>
                <hr style="border-color:#BBDEFB; margin:.6rem 0;">
                <div style="font-size:.82rem; color:#1565C0;">
                    Anomaly score: <b>{anomaly_score:.4f}</b><br>
                    (Semakin positif = semakin normal)<br><br>
                    Data pasien ini konsisten dengan pola umum dataset dan tidak menunjukkan 
                    nilai fitur yang ekstrem.
                </div>
            </div>""", unsafe_allow_html=True)
        st.caption("Metode: Isolation Forest (contamination=5%)")

    # ── Feature Gauge Chart ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"> Analisis Detail Input</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        # Radar / bar comparison
        fig, ax = plt.subplots(figsize=(9, 5))
        input_vals = [pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf * 100, age]
        ref_non_diab = df_clean[df_clean["Outcome"] == 0][FEATURE_NAMES].mean().values
        ref_diab     = df_clean[df_clean["Outcome"] == 1][FEATURE_NAMES].mean().values
        # normalise for display
        max_vals = df_clean[FEATURE_NAMES].max().values
        norm_input    = input_vals / np.array([20,300,150,100,900,80,300,100])
        norm_non_diab = ref_non_diab / df_clean[FEATURE_NAMES].max().values
        norm_diab     = ref_diab     / df_clean[FEATURE_NAMES].max().values

        short_names = ["Preg", "Gluc", "BP", "Skin", "Ins", "BMI", "DPF", "Age"]
        x = np.arange(len(short_names))
        w = 0.25
        ax.bar(x - w,   norm_non_diab, w, label="Rata-rata Non-Diabetes",
               color="#90CAF9", alpha=0.85)
        ax.bar(x,       norm_diab,     w, label="Rata-rata Diabetes",
               color="#EF9A9A", alpha=0.85)
        ax.bar(x + w,   norm_input,    w, label="Input Pasien",
               color="#1565C0", alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_ylabel("Nilai Ternormalisasi (0–1)", fontsize=10)
        ax.set_title("Perbandingan Input Pasien vs Rata-rata", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        # Probability gauge
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        sizes  = [probability[0], probability[1]]
        colors = ["#42A5F5", "#EF5350"]
        explode = (0, 0.05) if probability[1] > 0.5 else (0.05, 0)
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=["Tidak Diabetes", "Diabetes"],
            colors=colors, autopct='%1.1f%%', explode=explode,
            startangle=90, textprops={"fontsize": 11},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"}
        )
        for at in autotexts:
            at.set_fontsize(13)
            at.set_fontweight('bold')
        ax2.set_title("Probabilitas Prediksi\n(Random Forest)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Input summary table ────────────────────────────────────────────────────
    st.markdown("**Ringkasan Data Input**")
    reference = df_clean[FEATURE_NAMES].describe().loc[["mean", "min", "max"]].T
    summary_df = pd.DataFrame({
        "Fitur": FEATURE_NAMES,
        "Nilai Input": [pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, round(dpf, 3), age],
        "Rata-rata Dataset": reference["mean"].round(2).values,
        "Min Dataset": reference["min"].round(2).values,
        "Max Dataset": reference["max"].round(2).values,
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.warning(" **Disclaimer:** Hasil prediksi ini bersifat informatif dan tidak menggantikan diagnosis medis profesional. Selalu konsultasikan dengan dokter untuk keputusan klinis.")

else:
    st.info(" Silakan isi form di atas dan klik **Proses Prediksi** untuk melihat hasil analisis.")

    # Show example cases
    st.markdown('<div class="section-header"> Contoh Kasus</div>', unsafe_allow_html=True)
    examples = pd.DataFrame({
        "Kasus": ["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"],
        "Pregnancies": [1, 3, 8],
        "Glucose":     [85, 120, 180],
        "BloodPressure": [66, 72, 92],
        "SkinThickness": [29, 35, 45],
        "Insulin":     [0, 80, 200],
        "BMI":         [26.6, 33.6, 45.8],
        "DPF":         [0.351, 0.627, 0.878],
        "Age":         [25, 35, 50],
    })
    st.dataframe(examples, use_container_width=True, hide_index=True)
    st.caption("Isi form dengan salah satu contoh kasus di atas untuk mencobanya.")
