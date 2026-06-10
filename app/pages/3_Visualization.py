import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_models, preprocess, load_dataset, FEATURE_NAMES

st.set_page_config(page_title="Visualization", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .page-title  { font-size:2rem; font-weight:800; color:#1565C0; margin-bottom:.2rem; }
    .section-header {
        font-size:1.2rem; font-weight:700; color:#1565C0;
        border-left:4px solid #42A5F5; padding-left:.7rem; margin:1.5rem 0 .8rem 0;
    }
    .metric-box {
        background:white; border-radius:12px; border:1px solid #E3F2FD;
        padding:1rem; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,.05);
    }
    .metric-val { font-size:1.8rem; font-weight:800; color:#1565C0; }
    .metric-lbl { font-size:.8rem; color:#78909C; font-weight:600; }
    .result-title { font-size:1.6rem; font-weight:800; margin-bottom:.4rem; color:#37474F; }
    .result-sub   { font-size:.95rem; color:#546E7A; }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0D47A1,#1565C0); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("##  Diabetes")
    st.markdown("---")
    st.markdown("""
    <div style='color:white;'>
    <b>Menu:</b><br> Home<br> Dataset Overview<br>
     Prediction<br> <b>Visualization ←</b><br> About
    </div>""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=" Memuat model dan data...")
def load_all():
    return get_models()

sc, clf, km, iso, X_all, df_clean, best_k = load_all()

# Train/test split for evaluation
X = X_all[:, :8]
Y = df_clean['Outcome'].values.astype(int)
X_tr, X_te, Y_tr, Y_te = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

Y_pred   = clf.predict(X_te).astype(int)
Y_te     = Y_te.astype(int)
Y_proba  = clf.predict_proba(X_te)[:, 1]

# Cluster & anomaly labels for full dataset
X = np.nan_to_num(X, nan=0.0)
cluster_labels = km.predict(X)
anomaly_pred   = iso.predict(X)
anomaly_scores = iso.decision_function(X)
anomaly_labels = np.where(anomaly_pred == -1, 1, 0)

# PCA
pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X)
ev_r   = pca.explained_variance_ratio_

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title"> Visualization</div>', unsafe_allow_html=True)
st.markdown("Visualisasi hasil analisis ketiga metode machine learning secara lengkap.")

tabs = st.tabs([" Classification", " Clustering", " Anomaly Detection", " Perbandingan Model"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Evaluasi Model Random Forest</div>', unsafe_allow_html=True)

    # Metrics row
    Y_te   = Y_te.astype(int)
    Y_pred = clf.predict(X_te).astype(int)
    acc = accuracy_score(Y_te, Y_pred)
    pre = precision_score(Y_te, Y_pred)
    rec = recall_score(Y_te, Y_pred)
    f1  = f1_score(Y_te, Y_pred)
    fpr, tpr, _ = roc_curve(Y_te, Y_proba)
    roc_auc = auc(fpr, tpr)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (val, lbl) in zip(
        [m1, m2, m3, m4, m5],
        [(f"{acc*100:.2f}%","Accuracy"),
         (f"{pre:.4f}","Precision"),
         (f"{rec:.4f}","Recall"),
         (f"{f1:.4f}","F1 Score"),
         (f"{roc_auc:.4f}","AUC-ROC")]):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(Y_te, Y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Tidak Diabetes","Diabetes"],
                    yticklabels=["Tidak Diabetes","Diabetes"],
                    linewidths=1, cbar_kws={"shrink":.8})
        ax.set_title("Confusion Matrix — Random Forest", fontsize=13, fontweight='bold')
        ax.set_ylabel("Label Aktual", fontsize=11)
        ax.set_xlabel("Label Prediksi", fontsize=11)
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.22,
                f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
                ha='center', transform=ax.transAxes, fontsize=10, color='#546E7A')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ROC Curve
    with col2:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='#1565C0', lw=2.5,
                label=f'Random Forest (AUC = {roc_auc:.3f})')
        ax.plot([0,1],[0,1], color='#90A4AE', linestyle='--', lw=1.5, label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#1565C0')
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Feature Importance
    st.markdown("**Feature Importance**")
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_fi = ['#1565C0' if i == indices[0] else '#42A5F5' for i in range(len(FEATURE_NAMES))]
    bars = ax.bar(range(len(FEATURE_NAMES)), importances[indices],
                  color=colors_fi, edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=25, ha='right', fontsize=11)
    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Feature Importance — Random Forest\n(Biru Gelap = Fitur Paling Berpengaruh)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance Score', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Classification Report
    st.markdown("**Classification Report**")
    cr_dict = classification_report(Y_te, Y_pred,
                                    target_names=["Tidak Diabetes","Diabetes"],
                                    output_dict=True)
    cr_df = pd.DataFrame(cr_dict).T.round(4)
    st.dataframe(cr_df, use_container_width=True)
    
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil = silhouette_score(X, cluster_labels)
    db  = davies_bouldin_score(X, cluster_labels)

    st.markdown('<div class="section-header">Evaluasi K-Means Clustering</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{best_k}</div>
            <div class="metric-lbl">K Optimal</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{sil:.4f}</div>
            <div class="metric-lbl">Silhouette Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{db:.4f}</div>
            <div class="metric-lbl">Davies-Bouldin Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    cluster_colors = ['#2196F3', '#FF5722', '#4CAF50']

    # PCA Cluster Plot
    with col1:
        st.markdown("**Visualisasi Cluster (PCA 2D)**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        centroids_pca = pca.transform(km.cluster_centers_)
        for i in range(best_k):
            mask = cluster_labels == i
            axes[0].scatter(X_pca[mask,0], X_pca[mask,1],
                            c=cluster_colors[i % len(cluster_colors)],
                            label=f'Cluster {i}', alpha=0.6, s=30)
        axes[0].scatter(centroids_pca[:,0], centroids_pca[:,1],
                        c='black', marker='X', s=200, label='Centroid', zorder=5)
        axes[0].set_title('Cluster K-Means', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({ev_r[0]*100:.1f}%)', fontsize=10)
        axes[0].set_ylabel(f'PC2 ({ev_r[1]*100:.1f}%)', fontsize=10)
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        for outcome, color in {0:'#2196F3', 1:'#FF5722'}.items():
            mask = df_clean['Outcome'].values == outcome
            label = 'Tidak Diabetes' if outcome == 0 else 'Diabetes'
            axes[1].scatter(X_pca[mask,0], X_pca[mask,1],
                            c=color, label=label, alpha=0.6, s=30)
        axes[1].set_title('Label Asli (Outcome)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({ev_r[0]*100:.1f}%)', fontsize=10)
        axes[1].set_ylabel(f'PC2 ({ev_r[1]*100:.1f}%)', fontsize=10)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
        plt.suptitle('K-Means Clustering vs Label Asli', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Elbow Method
    with col2:
        st.markdown("**Elbow Method (WCSS)**")
        wcss = []
        k_range = range(2, 9)
        for k in k_range:
            km_tmp = __import__('sklearn.cluster', fromlist=['KMeans']).KMeans(
                n_clusters=k, random_state=42, n_init=10)
            km_tmp.fit(X)
            wcss.append(km_tmp.inertia_)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(list(k_range), wcss, 'o-', color='#1565C0', lw=2.5, ms=8)
        ax.axvline(best_k, color='#E53935', linestyle='--', lw=1.5,
                   label=f'K optimal = {best_k}')
        ax.fill_between(list(k_range), wcss, alpha=0.1, color='#1565C0')
        ax.set_xlabel('Jumlah Cluster (K)', fontsize=11)
        ax.set_ylabel('WCSS (Inertia)', fontsize=11)
        ax.set_title('Elbow Method', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Boxplot per cluster
    st.markdown("**Distribusi Fitur per Cluster**")
    df_c2 = df_clean.copy()
    df_c2["Cluster"] = cluster_labels
    features_plot = ['Glucose', 'BMI', 'Age', 'Insulin']
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for idx, feat in enumerate(features_plot):
        data_per = [df_c2[df_c2["Cluster"]==c][feat].values for c in range(best_k)]
        bp = axes[idx].boxplot(data_per, patch_artist=True,
                               labels=[f'C{i}' for i in range(best_k)])
        for patch, color in zip(bp['boxes'], cluster_colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        axes[idx].set_title(feat, fontsize=11, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].spines[['top','right']].set_visible(False)
    plt.suptitle('Karakteristik Fitur per Cluster', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Cluster stats table
    st.markdown("**Statistik Rata-rata per Cluster**")
    df_c2["Cluster"] = cluster_labels
    cluster_tbl = df_c2.groupby("Cluster")[FEATURE_NAMES + ["Outcome"]].mean().round(2)
    cluster_tbl.index = [f"Cluster {i}" for i in cluster_tbl.index]
    cluster_tbl.rename(columns={"Outcome": "% Diabetes"}, inplace=True)
    cluster_tbl["% Diabetes"] = (cluster_tbl["% Diabetes"] * 100).round(1).astype(str) + "%"
    st.dataframe(cluster_tbl, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    n_normal  = int(np.sum(anomaly_labels == 0))
    n_anomaly = int(np.sum(anomaly_labels == 1))

    st.markdown('<div class="section-header">Hasil Isolation Forest</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (val, lbl) in zip(
        [c1, c2, c3],
        [(len(anomaly_labels), "Total Data"),
         (f"{n_normal} ({n_normal/len(anomaly_labels)*100:.1f}%)", "Data Normal"),
         (f"{n_anomaly} ({n_anomaly/len(anomaly_labels)*100:.1f}%)", "Data Anomali")]):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Score distribution + PCA
    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(anomaly_scores[anomaly_labels==0], bins=30,
                     color='#2196F3', alpha=0.7, label='Normal', density=True)
        axes[0].hist(anomaly_scores[anomaly_labels==1], bins=30,
                     color='#FF5722', alpha=0.8, label='Anomali', density=True)
        axes[0].axvline(x=0, color='black', linestyle='--', lw=1.5, label='Batas (0)')
        axes[0].set_title('Distribusi Anomaly Score', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Anomaly Score'); axes[0].set_ylabel('Density')
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        axes[1].scatter(X_pca[anomaly_labels==0,0], X_pca[anomaly_labels==0,1],
                        c='#2196F3', alpha=0.5, s=25, label=f'Normal ({n_normal})')
        axes[1].scatter(X_pca[anomaly_labels==1,0], X_pca[anomaly_labels==1,1],
                        c='#FF5722', alpha=0.9, s=70, marker='X', label=f'Anomali ({n_anomaly})')
        axes[1].set_title('Anomali dalam PCA 2D', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({ev_r[0]*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({ev_r[1]*100:.1f}%)')
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
        plt.suptitle('Isolation Forest — Anomaly Detection', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        # Cross-tab anomaly vs diabetes
        st.markdown("**Hubungan Anomali dengan Diagnosis Diabetes**")
        df_anom = df_clean.copy()
        df_anom["Anomaly"] = anomaly_labels
        cross = pd.crosstab(
            df_anom["Anomaly"].map({0:"Normal",1:"Anomali"}),
            df_anom["Outcome"].map({0:"Tidak Diabetes",1:"Diabetes"}),
            margins=True)
        st.dataframe(cross, use_container_width=True)
        ad_rate = df_anom[df_anom["Anomaly"]==1]["Outcome"].mean()*100
        nd_rate = df_anom[df_anom["Anomaly"]==0]["Outcome"].mean()*100
        st.info(f"📊 Tingkat diabetes pada **data anomali**: **{ad_rate:.1f}%**  \n"
                f"📊 Tingkat diabetes pada **data normal**: **{nd_rate:.1f}%**")

    # Feature distribution: Normal vs Anomali
    st.markdown("**Distribusi Fitur: Normal vs Anomali**")
    features_anom = ['Glucose', 'Insulin', 'BMI', 'BloodPressure', 'SkinThickness', 'Age']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    df_anom2 = df_clean.copy()
    df_anom2["Anomaly"] = anomaly_labels
    for idx, feat in enumerate(features_anom):
        axes[idx].hist(df_anom2[df_anom2["Anomaly"]==0][feat], bins=25,
                       color='#2196F3', alpha=0.6, label='Normal', density=True)
        axes[idx].hist(df_anom2[df_anom2["Anomaly"]==1][feat], bins=25,
                       color='#FF5722', alpha=0.75, label='Anomali', density=True)
        axes[idx].set_title(f'Distribusi {feat}', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9); axes[idx].grid(alpha=0.3)
        axes[idx].spines[['top','right']].set_visible(False)
    plt.suptitle('Perbandingan Distribusi Fitur: Normal vs Anomali',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Top 10 anomalies
    st.markdown("**Top 10 Data Paling Anomali**")
    df_anom2["Anomaly_Score"] = anomaly_scores
    top10 = df_anom2.nsmallest(10, "Anomaly_Score")[
        ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age','Outcome','Anomaly_Score']
    ].round(2)
    top10["Outcome"] = top10["Outcome"].map({0:"Tidak Diabetes",1:"Diabetes"})
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    st.markdown('<div class="section-header">Perbandingan Semua Model Klasifikasi</div>', unsafe_allow_html=True)
    st.info(" Melakukan 5-Fold Cross Validation untuk semua model...")

    @st.cache_data(show_spinner=" Menjalankan cross-validation...")
    def run_cv(_X, _Y):
        _Y = _Y.astype(int) 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models = {
        'RF Default' : RandomForestClassifier(n_estimators=100, random_state=42),
        'RF Tuned'   : clf,
        'Log. Reg.'  : LogisticRegression(max_iter=1000, random_state=42),
        'SVM'        : SVC(kernel='rbf', probability=True, random_state=42),
        'Dec. Tree'  : DecisionTreeClassifier(random_state=42),
        }
        results = {}
        for name, model in models.items():
            try:
                scores = cross_val_score(model, _X, _Y.astype(int), cv=skf, scoring='accuracy')
                results[name] = {"mean": scores.mean()*100, "std": scores.std()*100,
                                 "min": scores.min()*100, "max": scores.max()*100}
            except Exception as e:
                results[name] = {"mean": None, "std": None, "min": None, "max": None}
                st.warning(f"{name} error: {e}")
        return results

    cv_results = run_cv(X, Y)
    cv_df = pd.DataFrame(cv_results).T.round(3)
    cv_df.index.name = "Model"
    cv_df.columns = ["Mean Acc (%)", "Std (%)", "Min (%)", "Max (%)"]
    cv_df = cv_df.sort_values("Mean Acc (%)", ascending=False)
    st.dataframe(cv_df, use_container_width=True)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean"] for n in names]
    stds   = [cv_results[n]["std"]  for n in names]
    colors_bar = ["#1565C0","#42A5F5","#90CAF9","#BBDEFB"]
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors_bar, edgecolor='white', linewidth=0.8,
                  error_kw={"elinewidth":1.5, "ecolor":"#455A64"})
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Akurasi (%)', fontsize=11)
    ax.set_title('Perbandingan Akurasi — 5-Fold Cross Validation', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Perbandingan performa detail
    st.markdown('<div class="section-header">Perbandingan Performa Semua Model Classification</div>', unsafe_allow_html=True)

    models_eval = {
        'RF Default' : RandomForestClassifier(n_estimators=100, random_state=42),
        'RF Tuned'   : clf,
        'Log. Reg.'  : LogisticRegression(max_iter=1000, random_state=42),
        'SVM'        : SVC(kernel='rbf', probability=True, random_state=42),
        'Dec. Tree'  : DecisionTreeClassifier(random_state=42),
    }

    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    comparison_data = []
    for name, model in models_eval.items():
        model.fit(X_tr, Y_tr.astype(int))
        y_pred = model.predict(X_te)
        comparison_data.append({
            'Model'    : name,
            'Accuracy' : accuracy_score(Y_te, y_pred) * 100,
            'Precision': precision_score(Y_te, y_pred),
            'Recall'   : recall_score(Y_te, y_pred),
            'F1 Score' : f1_score(Y_te, y_pred),
        })

    df_comparison = pd.DataFrame(comparison_data).set_index('Model')

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors_bar2 = ['#90CAF9', '#1565C0', '#A5D6A7', '#FFCC80', '#EF9A9A']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    ylabels = ['Accuracy (%)', 'Precision', 'Recall', 'F1 Score']

    for idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[idx // 2][idx % 2]
        values = df_comparison[metric].values
        bars = ax.bar(['RF Default','RF Tuned','Log. Reg.','SVM','Dec. Tree'],
                      values, color=colors_bar2, edgecolor='white', linewidth=1.5)
        best_idx = values.argmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        for bar, val in zip(bars, values):
            label = f'{val:.2f}%' if metric == 'Accuracy' else f'{val:.4f}'
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(0.3 if metric=='Accuracy' else 0.003),
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(f'Perbandingan {metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(['RF Default','RF Tuned','Log. Reg.','SVM','Dec. Tree'], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(min(values)*0.92, max(values)*1.05)

    plt.suptitle('Perbandingan Performa Semua Model Classification\n(Border Emas = Model Terbaik per Metrik)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Summary table all methods
    st.markdown('<div class="section-header">Ringkasan Semua Metode</div>', unsafe_allow_html=True)
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil2 = silhouette_score(X, cluster_labels)
    db2  = davies_bouldin_score(X, cluster_labels)
    n_anom = int(np.sum(anomaly_labels==1))
    summ_df = pd.DataFrame([
        ["Classification", "Random Forest", f"Accuracy: {acc*100:.2f}%  |  F1: {f1:.4f}  |  AUC: {roc_auc:.4f}"],
        ["Clustering",     "K-Means (k=3)", f"Silhouette: {sil2:.4f}  |  Davies-Bouldin: {db2:.4f}"],
        ["Anomaly Detection","Isolation Forest", f"Anomali: {n_anom} ({n_anom/len(anomaly_labels)*100:.1f}%)  |  Normal: {n_normal} ({n_normal/len(anomaly_labels)*100:.1f}%)"],
    ], columns=["Kategori", "Metode", "Hasil"])
    st.dataframe(summ_df, use_container_width=True, hide_index=True)
