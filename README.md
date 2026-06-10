# Diabetes Prediction System
**Data Mining -**

Sistem analisis prediksi diabetes berbasis Machine Learning menggunakan dataset Pima Indians Diabetes Database dengan tiga metode: **Klasifikasi (Random Forest)**, **Clustering (K-Means)**, dan **Anomaly Detection (Isolation Forest)**.

## Cara Menjalankan

### Prasyarat
- Python 3.9+
- pip

### Langkah-langkah

```bash
# 1. Masuk ke folder proyek
cd UAS_DataMining_KelompokDiabetes

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pastikan dataset ada
#    → Letakkan diabetes.csv di folder dataset/
#    → Download dari: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# 5. Jalankan aplikasi

py -m pip install streamlit
py -m streamlit run app/App.py

# 6. Buka browser
#    → http://localhost:8501
```

> **Catatan:** Jika `dataset/diabetes.csv` tidak ditemukan, aplikasi akan otomatis menggunakan data sampel sintetis yang merepresentasikan dataset asli.

---

##  Struktur Folder

```
UAS_DataMining_KelompokDiabetes/
│
├── dataset/
│   └── diabetes.csv
│
├── notebook/
│   └── Diabetes_prediction_final_New.ipynb
│
├── model/
│   ├── model_classification.pkl
│   ├── model_clustering.pkl
│   ├── model_anomaly.pkl
│   └── scaler.pkl
│
├── app/
│   ├── app.py                     ← Entry point (Home)
│   ├── utils.py                   ← Helper functions
│   ├── pages/
│   │   ├── 1_Dataset_Overview.py
│   │   ├── 2_Prediction.py
│   │   ├── 3_Visualization.py
│   │   └── 4_About.py
│   └── assets/
│
├── laporan/
│   └── laporan.pdf
│
├── requirements.txt
└── README.md
```
Catatan : Pastikan Struktur foldernya sama kecuali laporan
---

## 🧠 Metode yang Digunakan

| Metode | Algoritma | Tujuan |
|--------|-----------|--------|
| Classification | Random Forest (+ Hyperparameter Tuning) | Prediksi positif/negatif diabetes |
| Clustering | K-Means (k=3, Elbow Method) | Segmentasi profil pasien |
| Anomaly Detection | Isolation Forest (contamination=5%) | Deteksi data pasien ekstrem |

---

##  Dataset

- **Nama:** Pima Indians Diabetes Database
- **Sumber:** UCI ML Repository / Kaggle
- **Ukuran:** 768 data × 9 kolom
- **Target:** `Outcome` (0 = Tidak Diabetes, 1 = Diabetes)

---

##  Hasil Model

- **Random Forest Accuracy:** ~79%
- **Silhouette Score (K-Means):** ~0.15–0.20
- **Anomali Terdeteksi:** ~5% dari total data
