import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
    .page-title { font-size:2rem; font-weight:800; color:#1565C0; margin-bottom:.2rem; }
    .section-header {
        font-size:1.2rem; font-weight:700; color:#1565C0;
        border-left:4px solid #42A5F5; padding-left:.7rem; margin:1.5rem 0 .8rem 0;
    }
    .method-card {
        background:white; border-radius:16px; padding:1.8rem;
        border:1px solid #E3F2FD; box-shadow:0 2px 12px rgba(0,0,0,.06);
        margin-bottom:1rem;
        color: #1a1a1a;
    }
    .method-title { font-size:1.1rem; font-weight:700; margin-bottom:.6rem; }
    .info-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:.4rem; }
    .tag {
        display:inline-block; padding:3px 12px; border-radius:20px;
        font-size:.8rem; font-weight:700; margin:3px;
    }
    .tag-blue  { background:#E3F2FD; color:#1565C0; border:1px solid #90CAF9; }
    .tag-green { background:#E8F5E9; color:#2E7D32; border:1px solid #A5D6A7; }
    .tag-red   { background:#FCE4EC; color:#880E4F; border:1px solid #F48FB1; }
    .tag-orange{ background:#FFF3E0; color:#E65100; border:1px solid #FFCC80; }
    .project-card {
        background: linear-gradient(135deg,#E3F2FD,#F3E5F5);
        border-radius:16px; padding:2rem; margin-bottom:1rem;
        color: #1a1a1a;
    }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0D47A1,#1565C0); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("##  Diabetes")
    st.markdown("---")
    st.markdown("""
    <div style='color:white;'>
    <b>Menu:</b><br> Home<br> Dataset Overview<br>
     Prediction<br> Visualization<br> <b>About ←</b>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="page-title"> About Proyek</div>', unsafe_allow_html=True)
st.markdown("Informasi lengkap tentang metode, dataset, dan proyek ini.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Metode", " Dataset", " Informasi Proyek"])

# ═══════════════════════════════════════════════════════════
# Tab 1 — Metode
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Metode Machine Learning yang Digunakan</div>', unsafe_allow_html=True)

    # Random Forest
    st.markdown("""
    <div class="method-card">
        <div class="method-title">🟡 1. Random Forest (Classification)</div>
        <span class="tag tag-green">Supervised Learning</span>
        <span class="tag tag-blue">Ensemble Method</span>
        <span class="tag tag-blue">Binary Classification</span>
        <br><br>
        <b>Deskripsi:</b><br>
        Random Forest adalah algoritma ensemble berbasis <em>Decision Tree</em> yang membangun banyak
        pohon keputusan secara paralel menggunakan teknik <em>bagging</em> (Bootstrap Aggregating) dan
        menggabungkan hasilnya dengan voting mayoritas. Algoritma ini dipilih karena:
        <ul>
            <li>Robust terhadap overfitting dibanding single Decision Tree</li>
            <li>Mendukung analisis Feature Importance</li>
            <li>Tidak memerlukan scaling fitur (namun tetap dilakukan untuk konsistensi)</li>
            <li>Performa tinggi pada dataset tabular medis</li>
        </ul>
        <b>Konfigurasi:</b>
        <ul>
            <li><code>n_estimators = 100</code> — Jumlah pohon keputusan</li>
            <li><code>criterion = 'entropy'</code> — Kriteria pemisahan node</li>
            <li><code>random_state = 42</code> — Reproduktifitas hasil</li>
            <li>Hyperparameter tuning: <b>RandomizedSearchCV</b> (50 iterasi, 5-Fold Stratified CV)</li>
        </ul>
        <b>Preprocessing:</b> MinMaxScaler → Normalisasi fitur ke rentang [0, 1]<br>
        <b>Split Data:</b> 80% training — 20% testing (stratified)
    </div>
    """, unsafe_allow_html=True)

    # K-Means
    st.markdown("""
    <div class="method-card">
        <div class="method-title">🔵 2. K-Means Clustering</div>
        <span class="tag tag-orange">Unsupervised Learning</span>
        <span class="tag tag-blue">Clustering</span>
        <span class="tag tag-blue">Prototype-Based</span>
        <br><br>
        <b>Deskripsi:</b><br>
        K-Means adalah algoritma clustering yang membagi data menjadi K cluster berdasarkan
        jarak Euclidean ke centroid terdekat. Proses iteratif: (1) inisialisasi centroid,
        (2) assignment data ke centroid terdekat, (3) update centroid, diulang hingga konvergen.
        <ul>
            <li>Efisien dan scalable untuk dataset berukuran sedang</li>
            <li>Menghasilkan cluster yang mudah diinterpretasikan</li>
            <li>Cocok untuk segmentasi profil pasien</li>
        </ul>
        <b>Pemilihan K Optimal:</b>
        <ul>
            <li><b>Elbow Method</b> — Mencari titik siku pada grafik WCSS vs K</li>
            <li><b>Silhouette Score</b> — Mengukur kohesi dan separasi cluster (mendekati 1 = baik)</li>
            <li><b>Davies-Bouldin Score</b> — Mengukur kesamaan antar cluster (mendekati 0 = baik)</li>
        </ul>
        <b>K Optimal:</b> k = 3 cluster<br>
        <b>Visualisasi:</b> PCA 2D untuk proyeksi dimensi tinggi ke 2 komponen utama
    </div>
    """, unsafe_allow_html=True)

    # Isolation Forest
    st.markdown("""
    <div class="method-card">
        <div class="method-title">🔴 3. Isolation Forest (Anomaly Detection)</div>
        <span class="tag tag-red">Unsupervised Learning</span>
        <span class="tag tag-blue">Anomaly Detection</span>
        <span class="tag tag-blue">Ensemble Method</span>
        <br><br>
        <b>Deskripsi:</b><br>
        Isolation Forest mendeteksi anomali berdasarkan prinsip bahwa data anomali lebih mudah
        <em>"diisolasi"</em> dibandingkan data normal. Algoritma membangun banyak pohon isolasi
        secara random; data anomali memiliki <em>path length</em> yang lebih pendek menuju isolasi.
        <ul>
            <li>Efisien untuk dataset berdimensi tinggi</li>
            <li>Tidak memerlukan asumsi distribusi data</li>
            <li>Cocok untuk mendeteksi outlier medis atau error pencatatan data</li>
        </ul>
        <b>Konfigurasi:</b>
        <ul>
            <li><code>n_estimators = 100</code> — Jumlah pohon isolasi</li>
            <li><code>contamination = 0.05</code> — Estimasi 5% data adalah anomali</li>
            <li><code>max_samples = 'auto'</code> — Sampling otomatis</li>
        </ul>
        <b>Output:</b> 1 = Normal, -1 = Anomali; serta <em>anomaly score</em>
        (semakin negatif = semakin anomali)
    </div>
    """, unsafe_allow_html=True)

    # PCA
    st.markdown("""
    <div class="method-card">
        <div class="method-title">📐 4. PCA (Principal Component Analysis)</div>
        <span class="tag tag-blue">Dimensionality Reduction</span>
        <span class="tag tag-blue">Visualization</span>
        <br><br>
        PCA digunakan sebagai <b>teknik pendukung visualisasi</b>: memproyeksikan data 8 dimensi
        ke 2 komponen utama (PC1, PC2) yang menangkap variansi maksimum, sehingga hasil
        clustering dan anomaly detection dapat divisualisasikan dalam plot 2D.
        Variansi yang dijelaskan oleh kedua komponen ditampilkan di label sumbu.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Tab 2 — Dataset
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Informasi Dataset</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        ###  Pima Indians Diabetes Database

        Dataset ini berasal dari **National Institute of Diabetes and Digestive and Kidney Diseases**
        dan pertama kali dipublikasikan oleh Smith et al. (1988).

        **Tujuan asal:** Memprediksi apakah pasien menderita diabetes berdasarkan pengukuran diagnostik.

        **Karakteristik:**
        - **Sumber:** UCI ML Repository, Kaggle
        - **Populasi:** Wanita keturunan Indian Pima, Arizona, USA
        - **Usia minimal:** 21 tahun
        - **Total data:** 768 pasien
        - **Fitur:** 8 variabel numerik
        - **Target:** `Outcome` — 0 (Tidak Diabetes) / 1 (Diabetes)
        - **Distribusi kelas:** 65.1% negatif / 34.9% positif
        """)

    with col2:
        st.markdown("###  Deskripsi Fitur")
        feat_df = {
            "Fitur": [
                "Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age", "Outcome"
            ],
            "Deskripsi": [
                "Jumlah kehamilan",
                "Konsentrasi glukosa plasma (mg/dL), tes toleransi glukosa oral 2 jam",
                "Tekanan darah diastolik (mmHg)",
                "Ketebalan lipatan kulit trisep (mm)",
                "Kadar insulin serum 2 jam (μU/mL)",
                "Indeks Massa Tubuh = berat(kg)/tinggi²(m)",
                "Fungsi silsilah diabetes (pengaruh riwayat keluarga)",
                "Usia pasien (tahun)",
                "Variabel target: 1=Diabetes, 0=Tidak Diabetes"
            ],
            "Tipe": ["int","int","int","int","int","float","float","int","int"]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(feat_df), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Preprocessing yang Dilakukan</div>', unsafe_allow_html=True)
    st.markdown("""
    | Langkah | Detail |
    |---------|--------|
    | **1. Deteksi Missing Values** | Kolom Glucose, BloodPressure, SkinThickness, Insulin, BMI mengandung nilai 0 yang tidak valid secara medis |
    | **2. Imputasi** | Nilai 0 diganti dengan **mean** kolom masing-masing |
    | **3. Feature Scaling** | **MinMaxScaler** — normalisasi semua fitur ke rentang [0, 1] |
    | **4. Train-Test Split** | 80:20, stratified berdasarkan distribusi kelas `Outcome` |
    | **5. Stratified K-Fold** | 5-fold CV dengan StratifiedKFold untuk evaluasi yang adil |
    """)

    st.markdown('<div class="section-header">Referensi</div>', unsafe_allow_html=True)
    st.markdown("""
    - Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).
      *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus*.
      Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.
    - UCI Machine Learning Repository: [Pima Indians Diabetes Database](https://archive.ics.uci.edu/ml/datasets/diabetes)
    - Kaggle: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
    """)

# ═══════════════════════════════════════════════════════════
# Tab 3 — Informasi Proyek
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Informasi Umum Proyek</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="project-card">
        <b> Nama Proyek:</b> Diabetes Prediction System<br><br>
        <b> Mata Kuliah:</b> Data Mining<br><br>
        <b> Program Studi:</b> Sistem Informasi <br><br>
        <b> Semester:</b> Genap 2026<br><br>
        <b> Institusi:</b> Universitas Negeri Surabaya<br><br>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="project-card">
        <b> Tujuan Proyek:</b><br>
        Mengimplementasikan teknik Data Mining pada data medis diabetes
        menggunakan tiga pendekatan:
        <ol>
            <li>Klasifikasi (prediksi status diabetes)</li>
            <li>Clustering (segmentasi profil pasien)</li>
            <li>Anomaly Detection (identifikasi data ekstrem)</li>
        </ol>
        <b> Output:</b> Web aplikasi interaktif berbasis Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Struktur Folder Proyek</div>', unsafe_allow_html=True)
    st.code("""
UAS_DataMining_KelompokDiabetes/
│
├── dataset/
│   └── diabetes.csv                  ← Dataset utama
│
├── notebook/
│   └── Diabetes_prediction_final_New.ipynb   ← Notebook analisis
│
├── model/
│   ├── model_classification.pkl      ← Model Random Forest (tuned)
│   ├── model_clustering.pkl          ← Model K-Means
│   ├── model_anomaly.pkl             ← Model Isolation Forest
│   └── scaler.pkl                    ← MinMaxScaler
│
├── app/
│   ├── App.py                        ← Home page (entry point)
│   ├── utils.py                      ← Helper functions & model loader
│   ├── pages/
│   │   ├── 1_Dataset_Overview.py  ← Halaman dataset
│   │   ├── 2_Prediction.py        ← Halaman prediksi
│   │   ├── 3_Visualization.py     ← Halaman visualisasi
│   │   └── 4_About.py            ← Halaman about
│   └── assets/                       ← Gambar, ikon, dsb.
│
├── laporan/
│   └── laporan.pdf                   ← Laporan UAS
│
├── requirements.txt                  ← Daftar library Python
└── README.md                         ← Panduan menjalankan
    """, language="")

    st.markdown('<div class="section-header">Cara Menjalankan Aplikasi</div>', unsafe_allow_html=True)
    st.code("""
1. Clone atau download repository
cd UAS_DataMining_KelompokDiabetes

2. Install dependencies
py -m pip install -r requirements.txt
py -m pip install streamlit

3. Pastikan dataset ada di folder dataset/
#    → dataset/diabetes.csv

4. Jalankan aplikasi Streamlit
py -m streamlit run app/App.pyss

5. Buka browser: http://localhost:8501
    """, language="bash")

    st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        Python Libraries
        - `streamlit >= 1.32`
        - `pandas >= 2.0`
        - `numpy >= 1.24`
        - `scikit-learn >= 1.3`
        """)
    with col2:
        st.markdown("""
        Visualisasi
        - `matplotlib >= 3.7`
        - `seaborn >= 0.12`
        """)
    with col3:
        st.markdown("""
         Model Persistence
        - `pickle` (built-in)
        """)
