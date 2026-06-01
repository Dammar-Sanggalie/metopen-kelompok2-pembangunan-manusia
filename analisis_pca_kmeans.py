"""
=============================================================================
ANALISIS PCA + K-MEANS CLUSTERING - METOPEN KELOMPOK 2
Segmentasi dan Analisis Dinamika Pembangunan Manusia Provinsi di Indonesia
Periode 2015–2024
=============================================================================

Cara pakai:
    Pastikan file master_dataset.xlsx ada di data/processed/
    lalu jalankan: python analisis_pca_kmeans.py

Output yang dihasilkan (di folder output/):
    1. scree_plot.png          → untuk menentukan jumlah komponen PCA
    2. pca_biplot.png          → visualisasi sebaran provinsi di ruang PCA
    3. elbow_silhouette.png    → untuk menentukan jumlah klaster optimal
    4. cluster_scatter.png     → visualisasi klaster di ruang PCA
    5. cluster_heatmap.png     → profil rata-rata tiap klaster
    6. cluster_mobility.png    → dinamika perpindahan klaster 2015-2024
    7. hasil_cluster.xlsx      → master dataset lengkap dengan label klaster
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "data", "processed", "master_dataset.xlsx")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FITUR       = ["AHH", "RLS", "Persen_Miskin", "Persen_Keluhan"]
LABEL_FITUR = [
    "Angka Harapan Hidup",
    "Rata-rata Lama Sekolah",
    "% Penduduk Miskin",
    "% Keluhan Kesehatan",
]

# Warna klaster (maksimal 5 klaster)
WARNA_KLASTER = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

# =============================================================================
# STEP 1 — LOAD DATA & STANDARDISASI
# =============================================================================

def load_dan_standardisasi(filepath: str):
    print("\n" + "="*60)
    print("STEP 1: Load data dan standardisasi (Z-Score)...")
    print("="*60)

    df = pd.read_excel(filepath)
    print(f"  Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    X      = df[FITUR].values
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    print("  Statistik setelah standardisasi (mean ≈ 0, std ≈ 1):")
    df_std = pd.DataFrame(X_std, columns=FITUR)
    print(df_std.describe().round(4).to_string())

    return df, X, X_std, scaler


# =============================================================================
# STEP 2 — UJI KMO & BARTLETT (Validitas untuk PCA)
# =============================================================================

def uji_kmo_bartlett(X_std: np.ndarray):
    """
    KMO (Kaiser-Meyer-Olkin): mengukur kecukupan sampling.
        Nilai ≥ 0.5 → layak untuk PCA.
    Bartlett's Test of Sphericity: menguji apakah matriks korelasi
        bukan matriks identitas (variabel saling berkorelasi).
        p-value < 0.05 → layak untuk PCA.
    """
    print("\n" + "="*60)
    print("STEP 2: Uji KMO dan Bartlett's Test of Sphericity...")
    print("="*60)

    n, p = X_std.shape
    R    = np.corrcoef(X_std.T)   # matriks korelasi

    # --- Bartlett's Test ---
    det_R  = np.linalg.det(R)
    chi2   = -(n - 1 - (2 * p + 5) / 6) * np.log(det_R)
    df_chi = p * (p - 1) // 2
    p_val  = 1 - stats.chi2.cdf(chi2, df_chi)

    print(f"\n  Bartlett's Test of Sphericity:")
    print(f"    Chi-square : {chi2:.4f}")
    print(f"    df         : {df_chi}")
    print(f"    p-value    : {p_val:.6f}")
    if p_val < 0.05:
        print("    → ✓ SIGNIFIKAN (p < 0.05) — data layak untuk PCA")
    else:
        print("    → [PERINGATAN] p ≥ 0.05 — data mungkin tidak ideal untuk PCA")

    # --- KMO (Anti-Image Correlation Method) ---
    R_inv = np.linalg.inv(R)
    A     = np.zeros_like(R)
    for i in range(p):
        for j in range(p):
            A[i, j] = -R_inv[i, j] / np.sqrt(R_inv[i, i] * R_inv[j, j])
    np.fill_diagonal(A, 1.0)

    # KMO keseluruhan
    r2_sum = np.sum(R ** 2) - p               # jumlah r^2 off-diagonal
    a2_sum = np.sum(A ** 2) - p               # jumlah a^2 off-diagonal
    kmo    = r2_sum / (r2_sum + a2_sum)

    print(f"\n  KMO Measure of Sampling Adequacy:")
    print(f"    KMO = {kmo:.4f}")
    if kmo >= 0.8:
        ket = "Memuaskan (Marvelous)"
    elif kmo >= 0.7:
        ket = "Baik (Middling)"
    elif kmo >= 0.6:
        ket = "Cukup (Mediocre)"
    elif kmo >= 0.5:
        ket = "Dapat Diterima (Miserable)"
    else:
        ket = "Tidak Dapat Diterima"
    print(f"    Interpretasi: {ket}")
    if kmo >= 0.5:
        print("    → ✓ KMO ≥ 0.5 — data layak untuk PCA")
    else:
        print("    → [PERINGATAN] KMO < 0.5 — data kurang ideal untuk PCA")

    return kmo, chi2, df_chi, p_val


# =============================================================================
# STEP 3 — PCA & SCREE PLOT
# =============================================================================

def jalankan_pca(X_std: np.ndarray):
    print("\n" + "="*60)
    print("STEP 3: Principal Component Analysis (PCA)...")
    print("="*60)

    pca    = PCA()
    X_pca  = pca.fit_transform(X_std)

    eigenvalues  = pca.explained_variance_
    var_ratio    = pca.explained_variance_ratio_
    var_kumulatif = np.cumsum(var_ratio)
    loadings     = pca.components_.T  # shape: (n_fitur, n_komponen)

    print("\n  Eigenvalue dan Variansi per Komponen:")
    print(f"  {'PC':<5} {'Eigenvalue':>12} {'Var (%)':>10} {'Kumulatif (%)':>15}")
    print("  " + "-"*45)
    for i in range(len(eigenvalues)):
        bintang = " ← dipilih" if eigenvalues[i] >= 1.0 else ""
        print(f"  PC{i+1:<3} {eigenvalues[i]:>12.4f} {var_ratio[i]*100:>9.2f}% "
              f"{var_kumulatif[i]*100:>14.2f}%{bintang}")

    # Jumlah komponen dengan eigenvalue ≥ 1 (Kaiser criterion)
    n_komponen = np.sum(eigenvalues >= 1.0)
    print(f"\n  → Komponen dipilih (Eigenvalue ≥ 1): {n_komponen} komponen")
    print(f"  → Variansi yang dijelaskan: {var_kumulatif[n_komponen-1]*100:.2f}%")

    print("\n  Factor Loadings (korelasi variabel dengan komponen):")
    df_loadings = pd.DataFrame(
        loadings[:, :n_komponen],
        index=LABEL_FITUR,
        columns=[f"PC{i+1}" for i in range(n_komponen)]
    )
    print(df_loadings.round(4).to_string())

    # --- Scree Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Analisis Komponen Utama (PCA) — Scree Plot", fontsize=14, fontweight="bold")

    # Eigenvalue plot
    x = np.arange(1, len(eigenvalues) + 1)
    axes[0].bar(x, eigenvalues, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].plot(x, eigenvalues, "o-", color="#F44336", linewidth=2, markersize=8)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, label="Eigenvalue = 1")
    axes[0].set_xlabel("Komponen Utama", fontsize=11)
    axes[0].set_ylabel("Eigenvalue", fontsize=11)
    axes[0].set_title("Scree Plot — Eigenvalue", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"PC{i}" for i in x])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)

    for i, ev in enumerate(eigenvalues):
        axes[0].annotate(f"{ev:.3f}", (x[i], ev + 0.03), ha="center", fontsize=9)

    # Variansi kumulatif
    axes[1].bar(x, var_ratio * 100, color="#4CAF50", alpha=0.7, edgecolor="white",
                label="Variansi per komponen")
    axes[1].plot(x, var_kumulatif * 100, "s--", color="#FF9800", linewidth=2,
                 markersize=8, label="Variansi kumulatif")
    axes[1].axhline(y=80, color="gray", linestyle=":", linewidth=1.5, label="80% threshold")
    axes[1].set_xlabel("Komponen Utama", fontsize=11)
    axes[1].set_ylabel("Variansi yang Dijelaskan (%)", fontsize=11)
    axes[1].set_title("Variansi yang Dijelaskan", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"PC{i}" for i in x])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.4)

    for i, (vr, vk) in enumerate(zip(var_ratio, var_kumulatif)):
        axes[1].annotate(f"{vk*100:.1f}%", (x[i], vk * 100 + 1.5), ha="center", fontsize=9,
                         color="#FF9800", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scree_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Scree plot disimpan: {path}")

    # --- Biplot PCA ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, color="steelblue", s=30)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Biplot PCA — Sebaran Observasi di Ruang Komponen Utama", fontsize=13, fontweight="bold")

    # Gambar vektor loading
    skala = 3.0
    for i, (label, load) in enumerate(zip(LABEL_FITUR, loadings[:, :2])):
        ax.annotate("", xy=(load[0] * skala, load[1] * skala), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#F44336", lw=2))
        ax.text(load[0] * skala * 1.12, load[1] * skala * 1.12, label,
                ha="center", fontsize=9, color="#C62828", fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "pca_biplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Biplot PCA disimpan: {path}")

    # Ambil skor komponen untuk klasterisasi
    X_pca_2 = X_pca[:, :n_komponen]

    return pca, X_pca, X_pca_2, n_komponen, eigenvalues, var_ratio, var_kumulatif, df_loadings


# =============================================================================
# STEP 4 — OPTIMASI K-MEANS (Elbow + Silhouette)
# =============================================================================

def optimasi_kmeans(X_pca_2: np.ndarray):
    print("\n" + "="*60)
    print("STEP 4: Optimasi jumlah klaster K-Means...")
    print("="*60)

    K_range    = range(2, 8)
    wcss_list  = []
    sil_list   = []

    print(f"\n  {'K':<5} {'WCSS':>12} {'Silhouette':>12}")
    print("  " + "-"*32)

    for k in K_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca_2)
        wcss   = km.inertia_
        sil    = silhouette_score(X_pca_2, labels)
        wcss_list.append(wcss)
        sil_list.append(sil)
        print(f"  {k:<5} {wcss:>12.2f} {sil:>12.4f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Optimasi Jumlah Klaster K-Means", fontsize=14, fontweight="bold")

    K_list = list(K_range)

    axes[0].plot(K_list, wcss_list, "o-", color="#2196F3", linewidth=2.5, markersize=9)
    axes[0].set_xlabel("Jumlah Klaster (K)", fontsize=11)
    axes[0].set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
    axes[0].set_title("Elbow Method", fontsize=12)
    axes[0].set_xticks(K_list)
    axes[0].grid(alpha=0.4)
    for k, wcss in zip(K_list, wcss_list):
        axes[0].annotate(f"{wcss:.0f}", (k, wcss), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    axes[1].plot(K_list, sil_list, "s-", color="#4CAF50", linewidth=2.5, markersize=9)
    axes[1].set_xlabel("Jumlah Klaster (K)", fontsize=11)
    axes[1].set_ylabel("Silhouette Coefficient", fontsize=11)
    axes[1].set_title("Silhouette Coefficient", fontsize=12)
    axes[1].set_xticks(K_list)
    axes[1].grid(alpha=0.4)
    k_optimal = K_list[np.argmax(sil_list)]
    axes[1].axvline(k_optimal, color="#F44336", linestyle="--", linewidth=1.5,
                    label=f"K optimal = {k_optimal}")
    axes[1].legend()
    for k, sil in zip(K_list, sil_list):
        axes[1].annotate(f"{sil:.3f}", (k, sil), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "elbow_silhouette.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Plot Elbow + Silhouette disimpan: {path}")
    print(f"  → K optimal berdasarkan Silhouette: K = {k_optimal}")

    return k_optimal


# =============================================================================
# STEP 5 — KLASTERISASI FINAL & VISUALISASI
# =============================================================================

def klasterisasi_final(df: pd.DataFrame, X_std: np.ndarray,
                       X_pca: np.ndarray, X_pca_2: np.ndarray,
                       k_optimal: int, var_ratio: np.ndarray):
    print("\n" + "="*60)
    print(f"STEP 5: K-Means Clustering dengan K = {k_optimal}...")
    print("="*60)

    km     = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca_2)
    df     = df.copy()
    df["Klaster"] = labels + 1   # label mulai dari 1

    sil_final = silhouette_score(X_pca_2, labels)
    print(f"\n  Silhouette Coefficient final: {sil_final:.4f}")
    print(f"  WCSS final                  : {km.inertia_:.2f}")

    # Profil tiap klaster
    print("\n  Profil rata-rata per Klaster:")
    profil = df.groupby("Klaster")[FITUR].mean().round(4)
    profil["Jumlah_Observasi"] = df.groupby("Klaster")["Provinsi"].count()
    print(profil.to_string())

    # --- Scatter Plot Klaster ---
    fig, ax = plt.subplots(figsize=(11, 9))
    for k in sorted(df["Klaster"].unique()):
        mask = df["Klaster"] == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=WARNA_KLASTER[k - 1], label=f"Klaster {k}",
                   s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Centroid
    centroid_pca = km.cluster_centers_
    ax.scatter(centroid_pca[:, 0], centroid_pca[:, 1],
               c="black", marker="X", s=200, zorder=5, label="Centroid")

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax.set_title(f"K-Means Clustering (K={k_optimal}) di Ruang PCA\n"
                 f"Silhouette = {sil_final:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "cluster_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Scatter plot klaster disimpan: {path}")

    # --- Heatmap Profil Klaster ---
    profil_plot = df.groupby("Klaster")[FITUR].mean()
    profil_norm = (profil_plot - profil_plot.mean()) / profil_plot.std()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(profil_norm, annot=profil_plot.round(2), fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5, ax=ax,
                xticklabels=LABEL_FITUR,
                yticklabels=[f"Klaster {k}" for k in profil_plot.index])
    ax.set_title("Profil Rata-rata Indikator per Klaster\n"
                 "(nilai dalam sel = rata-rata asli, warna = z-score)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "cluster_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Heatmap profil klaster disimpan: {path}")

    return df, profil, sil_final


# =============================================================================
# STEP 6 — ANALISIS DINAMIKA PANEL (Cluster Mobility)
# =============================================================================

def analisis_mobilitas(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 6: Analisis Dinamika Panel (Cluster Mobility)...")
    print("="*60)

    tahun_list = sorted(df["Tahun"].unique())
    provinsi_list = sorted(df["Provinsi"].unique())

    # Pivot: baris=provinsi, kolom=tahun, nilai=klaster
    pivot = df.pivot(index="Provinsi", columns="Tahun", values="Klaster")

    # Hitung mobilitas per provinsi
    def hitung_mobilitas(row):
        changes = sum(row.iloc[i] != row.iloc[i+1] for i in range(len(row)-1))
        if changes == 0:
            return "Stabil"
        elif changes <= 2:
            return "Mobilitas Rendah"
        else:
            return "Mobilitas Tinggi"

    pivot["Mobilitas"] = pivot[tahun_list].apply(hitung_mobilitas, axis=1)

    print("\n  Distribusi mobilitas klaster:")
    print(pivot["Mobilitas"].value_counts().to_string())

    print("\n  Provinsi stabil (tidak pernah berganti klaster):")
    stabil = pivot[pivot["Mobilitas"] == "Stabil"].index.tolist()
    for p in stabil:
        klaster_nya = pivot.loc[p, tahun_list[0]]
        print(f"    {p} → Klaster {int(klaster_nya)} sepanjang 2015–2024")

    # --- Heatmap Mobilitas ---
    pivot_plot = pivot[tahun_list]

    n_klaster = df["Klaster"].nunique()
    cmap      = matplotlib.colors.ListedColormap(WARNA_KLASTER[:n_klaster])

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pivot_plot.values, aspect="auto", cmap=cmap,
                   vmin=0.5, vmax=n_klaster + 0.5)

    ax.set_xticks(range(len(tahun_list)))
    ax.set_xticklabels(tahun_list, fontsize=10)
    ax.set_yticks(range(len(provinsi_list)))
    ax.set_yticklabels(pivot_plot.index.tolist(), fontsize=8)
    ax.set_title("Dinamika Perpindahan Klaster Antarwaktu (2015–2024)\n"
                 "Tiap sel = nomor klaster provinsi di tahun tersebut",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tahun", fontsize=11)

    # Tulis nomor klaster di tiap sel
    for i in range(len(provinsi_list)):
        for j in range(len(tahun_list)):
            val = pivot_plot.iloc[i, j]
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=range(1, n_klaster + 1), shrink=0.6)
    cbar.set_label("Klaster", fontsize=10)
    cbar.set_ticklabels([f"Klaster {k}" for k in range(1, n_klaster + 1)])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cluster_mobility.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Heatmap mobilitas disimpan: {path}")

    # Matriks Transisi (dari tahun ke tahun berikutnya)
    n_k = df["Klaster"].nunique()
    print(f"\n  Matriks Transisi Klaster (rata-rata antar semua pasangan tahun berurutan):")
    total_transisi = np.zeros((n_k, n_k))
    count_transisi = 0

    for i in range(len(tahun_list) - 1):
        t1 = tahun_list[i]
        t2 = tahun_list[i + 1]
        for prov in provinsi_list:
            k1 = int(pivot.loc[prov, t1]) - 1
            k2 = int(pivot.loc[prov, t2]) - 1
            total_transisi[k1, k2] += 1
        count_transisi += 1

    mat = pd.DataFrame(
        total_transisi / count_transisi,
        index=[f"Dari K{k}" for k in range(1, n_k+1)],
        columns=[f"Ke K{k}" for k in range(1, n_k+1)]
    )
    print(mat.round(2).to_string())

    return pivot


# =============================================================================
# STEP 7 — SIMPAN HASIL & LAPORAN RINGKAS
# =============================================================================

def simpan_dan_laporan(df: pd.DataFrame, profil: pd.DataFrame,
                       kmo: float, chi2: float, df_chi: int, p_val: float,
                       eigenvalues, var_ratio, var_kumulatif,
                       df_loadings: pd.DataFrame, n_komponen: int,
                       k_optimal: int, sil_final: float):

    path_excel = os.path.join(OUTPUT_DIR, "hasil_cluster.xlsx")
    with pd.ExcelWriter(path_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data_Klaster", index=False)
        profil.to_excel(writer, sheet_name="Profil_Klaster")
        df_loadings.to_excel(writer, sheet_name="PCA_Loadings")

    print("\n" + "#"*60)
    print("#  LAPORAN HASIL ANALISIS — METOPEN KELOMPOK 2")
    print("#"*60)

    print(f"""
  ── Uji Validitas (KMO & Bartlett) ──────────────────────
  KMO                    : {kmo:.4f}  → layak PCA (≥ 0.5)
  Bartlett chi-square    : {chi2:.4f}
  Bartlett p-value       : {p_val:.6f}  → signifikan (< 0.05)

  ── Principal Component Analysis (PCA) ──────────────────
  Komponen terpilih (EV≥1): {n_komponen} komponen
  Variansi kumulatif      : {var_kumulatif[n_komponen-1]*100:.2f}%

  ── K-Means Clustering ───────────────────────────────────
  Jumlah klaster optimal  : K = {k_optimal}
  Silhouette Coefficient  : {sil_final:.4f}

  ── File Output ──────────────────────────────────────────
  output/scree_plot.png         → Scree plot PCA
  output/pca_biplot.png         → Biplot PCA
  output/elbow_silhouette.png   → Elbow + Silhouette
  output/cluster_scatter.png    → Scatter plot klaster
  output/cluster_heatmap.png    → Heatmap profil klaster
  output/cluster_mobility.png   → Dinamika klaster 2015–2024
  output/hasil_cluster.xlsx     → Data lengkap + klaster
    """)

    print("  ✓ Analisis selesai! Semua output tersimpan di folder output/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*60)
    print("#  ANALISIS PCA + K-MEANS — METOPEN KELOMPOK 2")
    print("#  Periode 2015–2024 | 34 Provinsi")
    print("#"*60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"[ERROR] File tidak ditemukan: {INPUT_FILE}\n"
            f"Jalankan preprocessing.py terlebih dahulu!"
        )

    df, X, X_std, scaler = load_dan_standardisasi(INPUT_FILE)

    kmo, chi2, df_chi, p_val = uji_kmo_bartlett(X_std)

    (pca, X_pca, X_pca_2, n_komponen,
     eigenvalues, var_ratio, var_kumulatif, df_loadings) = jalankan_pca(X_std)

    k_optimal = optimasi_kmeans(X_pca_2)

    df, profil, sil_final = klasterisasi_final(
        df, X_std, X_pca, X_pca_2, k_optimal, var_ratio
    )

    pivot_mobilitas = analisis_mobilitas(df)

    simpan_dan_laporan(
        df, profil, kmo, chi2, df_chi, p_val,
        eigenvalues, var_ratio, var_kumulatif,
        df_loadings, n_komponen, k_optimal, sil_final
    )

    return df


if __name__ == "__main__":
    df_hasil = main()