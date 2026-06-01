"""
=============================================================================
PREPROCESSING DATA - METOPEN KELOMPOK 2
Segmentasi dan Analisis Dinamika Pembangunan Manusia Provinsi di Indonesia
Periode 2015–2024
=============================================================================

Struktur folder yang dibutuhkan:
    project_metopen/
    ├── data/
    │   ├── raw/
    │   │   ├── AHH/          ← file AHH_2015.xlsx s/d AHH_2024.xlsx
    │   │   ├── RLS/          ← file RLS_2015.xlsx s/d RLS_2024.xlsx
    │   │   ├── Kemiskinan/   ← file Kemiskinan_2015.xlsx s/d Kemiskinan_2024.xlsx
    │   │   └── Kesehatan/    ← 1 file PERSEN_1.XLS
    │   └── processed/        ← hasil master dataset disimpan di sini
    └── preprocessing.py      ← file ini

Cara pakai:
    Sesuaikan PATH_RAW dan nama file di bagian KONFIGURASI di bawah,
    lalu jalankan: python preprocessing.py
=============================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI — Sesuaikan path dan nama file kamu di sini
# =============================================================================

# Path folder utama (sesuaikan dengan lokasi project kamu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_AHH        = os.path.join(BASE_DIR, "data", "raw", "AHH")
PATH_RLS        = os.path.join(BASE_DIR, "data", "raw", "RLS")
PATH_KEMISKINAN = os.path.join(BASE_DIR, "data", "raw", "Kemiskinan")
PATH_KESEHATAN  = os.path.join(BASE_DIR, "data", "raw", "Kesehatan", "PERSEN_1.XLS")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAHUN = list(range(2015, 2025))  # 2015 s/d 2024

# 34 provinsi definitif yang digunakan (sebelum pemekaran Papua)
# Provinsi pemekaran (Papua Barat Daya, Papua Selatan, Papua Tengah,
# Papua Pegunungan) TIDAK dimasukkan karena tidak punya data 2015–2021
PROVINSI_VALID = [
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI",
    "SUMATERA SELATAN", "BENGKULU", "LAMPUNG", "KEP. BANGKA BELITUNG",
    "KEP. RIAU", "DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH",
    "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN", "BALI",
    "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT",
    "KALIMANTAN TENGAH", "KALIMANTAN SELATAN", "KALIMANTAN TIMUR",
    "KALIMANTAN UTARA", "SULAWESI UTARA", "SULAWESI TENGAH",
    "SULAWESI SELATAN", "SULAWESI TENGGARA", "GORONTALO",
    "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", "PAPUA BARAT", "PAPUA",
]

# =============================================================================
# FUNGSI UTILITAS
# =============================================================================

def standarisasi_nama_provinsi(nama: str) -> str:
    """
    Menyeragamkan nama provinsi ke format huruf kapital semua,
    dan menangani variasi penulisan yang umum ditemukan di data BPS.
    """
    if not isinstance(nama, str):
        return ""
    
    nama = nama.strip().upper()
    
    # Mapping variasi nama → nama standar
    mapping = {
        "ACEH"                          : "ACEH",
        "D.I. ACEH"                     : "ACEH",
        "SUMATERA UTARA"                : "SUMATERA UTARA",
        "SUMATRA UTARA"                 : "SUMATERA UTARA",
        "SUMATERA BARAT"                : "SUMATERA BARAT",
        "SUMATRA BARAT"                 : "SUMATERA BARAT",
        "RIAU"                          : "RIAU",
        "JAMBI"                         : "JAMBI",
        "SUMATERA SELATAN"              : "SUMATERA SELATAN",
        "SUMATRA SELATAN"               : "SUMATERA SELATAN",
        "BENGKULU"                      : "BENGKULU",
        "LAMPUNG"                       : "LAMPUNG",
        "KEP. BANGKA BELITUNG"          : "KEP. BANGKA BELITUNG",
        "KEPULAUAN BANGKA BELITUNG"     : "KEP. BANGKA BELITUNG",
        "BANGKA BELITUNG"               : "KEP. BANGKA BELITUNG",
        "KEP. RIAU"                     : "KEP. RIAU",
        "KEPULAUAN RIAU"                : "KEP. RIAU",
        "DKI JAKARTA"                   : "DKI JAKARTA",
        "JAKARTA"                       : "DKI JAKARTA",
        "JAWA BARAT"                    : "JAWA BARAT",
        "JAWA TENGAH"                   : "JAWA TENGAH",
        "DI YOGYAKARTA"                 : "DI YOGYAKARTA",
        "D.I. YOGYAKARTA"               : "DI YOGYAKARTA",
        "D.I YOGYAKARTA"                : "DI YOGYAKARTA",
        "YOGYAKARTA"                    : "DI YOGYAKARTA",
        "JAWA TIMUR"                    : "JAWA TIMUR",
        "BANTEN"                        : "BANTEN",
        "BALI"                          : "BALI",
        "NUSA TENGGARA BARAT"           : "NUSA TENGGARA BARAT",
        "NUSA TENGGARA TIMUR"           : "NUSA TENGGARA TIMUR",
        "KALIMANTAN BARAT"              : "KALIMANTAN BARAT",
        "KALIMANTAN TENGAH"             : "KALIMANTAN TENGAH",
        "KALIMANTAN SELATAN"            : "KALIMANTAN SELATAN",
        "KALIMANTAN TIMUR"              : "KALIMANTAN TIMUR",
        "KALIMANTAN UTARA"              : "KALIMANTAN UTARA",
        "SULAWESI UTARA"                : "SULAWESI UTARA",
        "SULAWESI TENGAH"               : "SULAWESI TENGAH",
        "SULAWESI SELATAN"              : "SULAWESI SELATAN",
        "SULAWESI TENGGARA"             : "SULAWESI TENGGARA",
        "GORONTALO"                     : "GORONTALO",
        "SULAWESI BARAT"                : "SULAWESI BARAT",
        "MALUKU"                        : "MALUKU",
        "MALUKU UTARA"                  : "MALUKU UTARA",
        "PAPUA BARAT"                   : "PAPUA BARAT",
        "PAPUA"                         : "PAPUA",
        # Provinsi baru (pemekaran) — tidak dimasukkan ke PROVINSI_VALID
        "PAPUA BARAT DAYA"              : "PAPUA BARAT DAYA",
        "PAPUA SELATAN"                 : "PAPUA SELATAN",
        "PAPUA TENGAH"                  : "PAPUA TENGAH",
        "PAPUA PEGUNUNGAN"              : "PAPUA PEGUNUNGAN",
    }
    
    return mapping.get(nama, nama)


def cek_file_ada(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] File {label} tidak ditemukan di:\n  {path}\n"
            f"Pastikan path dan nama file sudah sesuai di bagian KONFIGURASI."
        )


# =============================================================================
# STEP 1 — LOAD & BERSIHKAN DATA AHH
# =============================================================================

def load_ahh(path_folder: str, tahun_list: list) -> pd.DataFrame:
    """
    Membaca semua file AHH tahunan, mengambil rata-rata AHH
    (Laki-laki + Perempuan) / 2, dan menggabungkannya ke format panel.

    Format file AHH BPS:
        Row 0 : label '38 Provinsi'
        Row 1 : judul indikator
        Row 2 : header jenis kelamin (Laki-laki | Perempuan)
        Row 3 : tahun
        Row 4+ : data provinsi
    
    Nilai '-' menandakan provinsi baru yang belum punya data → di-drop.
    """
    print("\n" + "="*60)
    print("STEP 1: Memproses data Angka Harapan Hidup (AHH)...")
    print("="*60)
    
    records = []
    
    for tahun in tahun_list:
        # Cari file dengan pola nama yang fleksibel
        pola = os.path.join(path_folder, f"*{tahun}*.xlsx")
        files = glob.glob(pola)
        
        if not files:
            # Coba ekstensi xls
            pola = os.path.join(path_folder, f"*{tahun}*.xls")
            files = glob.glob(pola)
        
        if not files:
            print(f"  [PERINGATAN] File AHH tahun {tahun} tidak ditemukan, dilewati.")
            continue
        
        filepath = files[0]
        print(f"  Membaca: {os.path.basename(filepath)}")
        
        df = pd.read_excel(filepath, header=None)
        
        # Data mulai dari baris ke-4 (index 4)
        # Kolom: 0=Provinsi, 1=Laki-laki, 2=Perempuan
        data = df.iloc[4:, :3].copy()
        data.columns = ["Provinsi", "AHH_L", "AHH_P"]
        data = data.reset_index(drop=True)
        
        for _, row in data.iterrows():
            provinsi_raw = row["Provinsi"]
            
            # Skip jika bukan nama provinsi
            if not isinstance(provinsi_raw, str) or provinsi_raw.strip() == "":
                continue
            
            # Skip jika nilai adalah '-' (provinsi baru belum ada data)
            if row["AHH_L"] == "-" or row["AHH_P"] == "-":
                continue
            
            # Konversi ke numerik, skip jika gagal
            try:
                ahh_l = float(row["AHH_L"])
                ahh_p = float(row["AHH_P"])
            except (ValueError, TypeError):
                continue
            
            if np.isnan(ahh_l) or np.isnan(ahh_p):
                continue
            
            nama_std = standarisasi_nama_provinsi(provinsi_raw)
            
            if nama_std not in PROVINSI_VALID:
                continue
            
            records.append({
                "Provinsi": nama_std,
                "Tahun"   : tahun,
                "AHH"     : round((ahh_l + ahh_p) / 2, 4),
            })
    
    df_ahh = pd.DataFrame(records)
    print(f"  → Total baris AHH: {len(df_ahh)} "
          f"({df_ahh['Provinsi'].nunique()} provinsi x {df_ahh['Tahun'].nunique()} tahun)")
    return df_ahh


# =============================================================================
# STEP 2 — LOAD & BERSIHKAN DATA RLS
# =============================================================================

def load_rls(path_folder: str, tahun_list: list) -> pd.DataFrame:
    """
    Membaca semua file RLS tahunan.

    Format file RLS BPS:
        Row 0 : label '38 Provinsi'
        Row 1 : judul indikator
        Row 2 : tahun
        Row 3+ : data provinsi (Kolom 0=Provinsi, Kolom 1=nilai RLS)
    
    Nilai '-' menandakan data tidak tersedia → di-drop.
    """
    print("\n" + "="*60)
    print("STEP 2: Memproses data Rata-Rata Lama Sekolah (RLS)...")
    print("="*60)
    
    records = []
    
    for tahun in tahun_list:
        pola = os.path.join(path_folder, f"*{tahun}*.xlsx")
        files = glob.glob(pola)
        
        if not files:
            pola = os.path.join(path_folder, f"*{tahun}*.xls")
            files = glob.glob(pola)
        
        if not files:
            print(f"  [PERINGATAN] File RLS tahun {tahun} tidak ditemukan, dilewati.")
            continue
        
        filepath = files[0]
        print(f"  Membaca: {os.path.basename(filepath)}")
        
        df = pd.read_excel(filepath, header=None)
        
        # Data mulai baris ke-3 (index 3)
        data = df.iloc[3:, :2].copy()
        data.columns = ["Provinsi", "RLS"]
        data = data.reset_index(drop=True)
        
        for _, row in data.iterrows():
            provinsi_raw = row["Provinsi"]
            
            if not isinstance(provinsi_raw, str) or provinsi_raw.strip() == "":
                continue
            
            if row["RLS"] == "-":
                continue
            
            try:
                rls_val = float(row["RLS"])
            except (ValueError, TypeError):
                continue
            
            if np.isnan(rls_val):
                continue
            
            nama_std = standarisasi_nama_provinsi(provinsi_raw)
            
            if nama_std not in PROVINSI_VALID:
                continue
            
            records.append({
                "Provinsi": nama_std,
                "Tahun"   : tahun,
                "RLS"     : round(rls_val, 4),
            })
    
    df_rls = pd.DataFrame(records)
    print(f"  → Total baris RLS: {len(df_rls)} "
          f"({df_rls['Provinsi'].nunique()} provinsi x {df_rls['Tahun'].nunique()} tahun)")
    return df_rls


# =============================================================================
# STEP 3 — LOAD & BERSIHKAN DATA KEMISKINAN
# =============================================================================

def load_kemiskinan(path_folder: str, tahun_list: list) -> pd.DataFrame:
    """
    Membaca semua file Kemiskinan tahunan.

    Strategi pemilihan kolom (otomatis):
    - Prioritas 1: 'Persentase Penduduk Miskin - September' jika nilainya angka
    - Prioritas 2: 'Persentase Penduduk Miskin - Maret' sebagai fallback
    
    Alasan: File BPS 2015–2018 menyediakan data September, sedangkan
    file 2019–2024 hanya menyediakan data Maret (kolom September berisi '...').
    Kedua periode pengukuran sama-sama valid sebagai representasi tahunan.

    Format file Kemiskinan BPS:
        Row 0 : header kolom
        Row 1+ : data provinsi
        Baris terakhir : 'Indonesia' (total nasional) → di-drop
    """
    print("\n" + "="*60)
    print("STEP 3: Memproses data Persentase Penduduk Miskin...")
    print("="*60)
    
    records = []
    
    for tahun in tahun_list:
        pola = os.path.join(path_folder, f"*{tahun}*.xlsx")
        files = glob.glob(pola)
        
        if not files:
            pola = os.path.join(path_folder, f"*{tahun}*.xls")
            files = glob.glob(pola)
        
        if not files:
            print(f"  [PERINGATAN] File Kemiskinan tahun {tahun} tidak ditemukan, dilewati.")
            continue
        
        filepath = files[0]
        
        df = pd.read_excel(filepath, header=0)
        kolom_provinsi = df.columns[0]

        # --- Deteksi otomatis kolom persentase yang berisi angka ---
        # Cari kolom September dan Maret berdasarkan nama
        kolom_sept  = next((c for c in df.columns if "september" in str(c).lower()
                            and "persen" in str(c).lower()), None)
        kolom_maret = next((c for c in df.columns if "maret" in str(c).lower()
                            and "persen" in str(c).lower()), None)

        # Cek apakah kolom September punya nilai angka (bukan '...')
        # dengan mencoba konversi baris pertama data
        def kolom_punya_angka(kolom):
            if kolom is None:
                return False
            try:
                val = df[kolom].iloc[0]
                float(val)
                return True
            except (ValueError, TypeError):
                return False

        if kolom_punya_angka(kolom_sept):
            kolom_persen = kolom_sept
            periode      = "September"
        elif kolom_punya_angka(kolom_maret):
            kolom_persen = kolom_maret
            periode      = "Maret"
        else:
            # Fallback terakhir: cari kolom persentase apapun yang berisi angka
            kolom_persen = None
            periode      = "?"
            for col in df.columns:
                if "persen" in str(col).lower() and kolom_punya_angka(col):
                    kolom_persen = col
                    periode      = str(col)
                    break

        if kolom_persen is None:
            print(f"  [PERINGATAN] Tidak dapat menemukan kolom persentase untuk tahun {tahun}, dilewati.")
            continue

        print(f"  Membaca: {os.path.basename(filepath)}  →  pakai kolom [{periode}]")
        
        for _, row in df.iterrows():
            provinsi_raw = row[kolom_provinsi]
            
            if not isinstance(provinsi_raw, str) or provinsi_raw.strip() == "":
                continue
            
            # Skip baris total Indonesia
            if "indonesia" in provinsi_raw.lower():
                continue
            
            try:
                persen_val = float(row[kolom_persen])
            except (ValueError, TypeError):
                continue
            
            if np.isnan(persen_val):
                continue
            
            nama_std = standarisasi_nama_provinsi(provinsi_raw)
            
            if nama_std not in PROVINSI_VALID:
                continue
            
            records.append({
                "Provinsi"      : nama_std,
                "Tahun"         : tahun,
                "Persen_Miskin" : round(persen_val, 4),
            })
    
    df_miskin = pd.DataFrame(records)
    print(f"  → Total baris Kemiskinan: {len(df_miskin)} "
          f"({df_miskin['Provinsi'].nunique()} provinsi x {df_miskin['Tahun'].nunique()} tahun)")
    return df_miskin


# =============================================================================
# STEP 4 — LOAD & BERSIHKAN DATA KELUHAN KESEHATAN
# =============================================================================

def load_keluhan_kesehatan(filepath: str, tahun_list: list) -> pd.DataFrame:
    """
    Membaca file Keluhan Kesehatan multi-tahun (2009–2025).
    Mengambil nilai 'Perkotaan + Perdesaan' yang merupakan gabungan
    Laki-laki dan Perempuan (rata-rata L+P) untuk tahun 2015–2024.

    Struktur kolom per tahun (6 kolom):
        tahun_start+0 : Laki-laki Perkotaan
        tahun_start+1 : Laki-laki Perdesaan
        tahun_start+2 : Laki-laki Perkotaan+Perdesaan  ← kita ambil ini
        tahun_start+3 : Perempuan Perkotaan
        tahun_start+4 : Perempuan Perdesaan
        tahun_start+5 : Perempuan Perkotaan+Perdesaan  ← dan ini

    Nilai yang kita gunakan = rata-rata(kolom L gabungan, kolom P gabungan)
    """
    print("\n" + "="*60)
    print("STEP 4: Memproses data Keluhan Kesehatan...")
    print("="*60)
    print(f"  Membaca: {os.path.basename(filepath)}")
    
    cek_file_ada(filepath, "Keluhan Kesehatan")
    
    # Deteksi engine berdasarkan ekstensi
    ext = os.path.splitext(filepath)[1].lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    
    df_raw = pd.read_excel(filepath, engine=engine, header=None)
    
    # Baris 2 berisi tahun, baris 4 berisi tipe daerah
    # Data provinsi mulai baris 5
    row_tahun    = list(df_raw.iloc[2])
    
    # Buat peta: tahun → posisi kolom awal
    year_col_start = {}
    for i, val in enumerate(row_tahun):
        if isinstance(val, (int, float)) and not pd.isna(val):
            year_col_start[int(val)] = i
    
    records = []
    
    # Data provinsi dari baris 5 ke atas (sebelum baris INDONESIA dan Sumber)
    data_rows = df_raw.iloc[5:, :]
    
    for row_idx, row in data_rows.iterrows():
        provinsi_raw = row.iloc[0]
        
        # Stop jika baris adalah INDONESIA atau baris sumber
        if not isinstance(provinsi_raw, str) or provinsi_raw.strip() == "":
            continue
        if "indonesia" in provinsi_raw.lower():
            continue
        if "sumber" in provinsi_raw.lower():
            continue
        
        nama_std = standarisasi_nama_provinsi(provinsi_raw)
        
        if nama_std not in PROVINSI_VALID:
            continue
        
        for tahun in tahun_list:
            if tahun not in year_col_start:
                continue
            
            start = year_col_start[tahun]
            
            # Kolom +2 = L Perkotaan+Perdesaan
            # Kolom +5 = P Perkotaan+Perdesaan
            try:
                val_l = row.iloc[start + 2]
                val_p = row.iloc[start + 5]
                
                # Handle nilai '-' atau string kosong
                if val_l == "-" or val_p == "-":
                    continue
                
                val_l = float(val_l)
                val_p = float(val_p)
                
                if np.isnan(val_l) or np.isnan(val_p):
                    continue
                
                # Rata-rata L+P
                keluhan_val = round((val_l + val_p) / 2, 4)
                
            except (ValueError, TypeError, IndexError):
                continue
            
            records.append({
                "Provinsi"       : nama_std,
                "Tahun"          : tahun,
                "Persen_Keluhan" : keluhan_val,
            })
    
    df_keluhan = pd.DataFrame(records)
    print(f"  → Total baris Keluhan Kesehatan: {len(df_keluhan)} "
          f"({df_keluhan['Provinsi'].nunique()} provinsi x {df_keluhan['Tahun'].nunique()} tahun)")
    return df_keluhan


# =============================================================================
# STEP 5 — GABUNGKAN SEMUA INDIKATOR KE MASTER DATASET
# =============================================================================

def gabungkan_master(df_ahh, df_rls, df_miskin, df_keluhan) -> pd.DataFrame:
    """
    Menggabungkan semua 4 indikator berdasarkan kunci (Provinsi, Tahun).
    Menggunakan inner join agar hanya baris yang lengkap (semua indikator
    tersedia) yang masuk ke master dataset.
    """
    print("\n" + "="*60)
    print("STEP 5: Menggabungkan semua indikator ke Master Dataset...")
    print("="*60)
    
    master = df_ahh.merge(df_rls,    on=["Provinsi", "Tahun"], how="inner")
    master = master.merge(df_miskin, on=["Provinsi", "Tahun"], how="inner")
    master = master.merge(df_keluhan,on=["Provinsi", "Tahun"], how="inner")
    
    # Urutkan berdasarkan Provinsi dan Tahun
    master = master.sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)
    
    return master


# =============================================================================
# STEP 6 — VALIDASI & LAPORAN KUALITAS DATA
# =============================================================================

def validasi_dataset(master: pd.DataFrame):
    """
    Menjalankan pengecekan validitas dan reliabilitas data:
    1. Cek kelengkapan (missing values)
    2. Cek jumlah provinsi dan tahun
    3. Cek outlier sederhana (nilai di luar rentang wajar)
    4. Statistik deskriptif
    """
    print("\n" + "="*60)
    print("STEP 6: Validasi dan Pengecekan Kualitas Data")
    print("="*60)
    
    n_provinsi = master["Provinsi"].nunique()
    n_tahun    = master["Tahun"].nunique()
    n_baris    = len(master)
    expected   = n_provinsi * n_tahun
    
    print(f"\n  Jumlah provinsi  : {n_provinsi}")
    print(f"  Rentang tahun    : {master['Tahun'].min()} – {master['Tahun'].max()}")
    print(f"  Jumlah tahun     : {n_tahun}")
    print(f"  Total baris      : {n_baris}")
    print(f"  Ekspektasi baris : {expected}")
    print(f"  Kelengkapan panel: {n_baris/expected*100:.1f}%")
    
    # Cek missing values
    print("\n  === Pengecekan Missing Values ===")
    missing = master.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ Tidak ada missing values!")
    else:
        print(missing[missing > 0])
    
    # Cek provinsi yang tidak lengkap (tidak punya data untuk semua tahun)
    print("\n  === Kelengkapan Per Provinsi ===")
    count_per_provinsi = master.groupby("Provinsi")["Tahun"].count()
    tidak_lengkap = count_per_provinsi[count_per_provinsi < n_tahun]
    if len(tidak_lengkap) == 0:
        print(f"  ✓ Semua {n_provinsi} provinsi memiliki data lengkap ({n_tahun} tahun)!")
    else:
        print("  [PERINGATAN] Provinsi dengan data tidak lengkap:")
        for prov, cnt in tidak_lengkap.items():
            print(f"    {prov}: hanya {cnt} tahun")
    
    # Rentang nilai wajar untuk validasi
    batas_wajar = {
        "AHH"           : (50, 85),   # tahun
        "RLS"           : (3, 15),    # tahun
        "Persen_Miskin" : (0, 50),    # persen
        "Persen_Keluhan": (5, 60),    # persen
    }
    
    print("\n  === Pengecekan Nilai di Luar Rentang Wajar ===")
    ada_outlier = False
    for kolom, (batas_min, batas_max) in batas_wajar.items():
        outliers = master[(master[kolom] < batas_min) | (master[kolom] > batas_max)]
        if len(outliers) > 0:
            ada_outlier = True
            print(f"  [PERINGATAN] {kolom}: {len(outliers)} nilai di luar [{batas_min}, {batas_max}]")
            print(outliers[["Provinsi", "Tahun", kolom]].to_string(index=False))
    if not ada_outlier:
        print("  ✓ Semua nilai berada dalam rentang yang wajar!")
    
    # Statistik deskriptif
    print("\n  === Statistik Deskriptif ===")
    desc = master[["AHH", "RLS", "Persen_Miskin", "Persen_Keluhan"]].describe()
    desc = desc.round(4)
    print(desc.to_string())
    
    return missing.sum() == 0 and len(tidak_lengkap) == 0


# =============================================================================
# MAIN — Jalankan semua step
# =============================================================================

def main():
    print("\n" + "#"*60)
    print("#  PREPROCESSING MASTER DATASET - METOPEN KELOMPOK 2")
    print("#  Periode 2015–2024 | 34 Provinsi")
    print("#"*60)
    
    # Cek ketersediaan folder
    for path, label in [(PATH_AHH, "AHH"), (PATH_RLS, "RLS"),
                         (PATH_KEMISKINAN, "Kemiskinan")]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"[ERROR] Folder {label} tidak ditemukan: {path}\n"
                f"Pastikan struktur folder sudah sesuai."
            )
    cek_file_ada(PATH_KESEHATAN, "Keluhan Kesehatan")
    
    # Jalankan tiap step
    df_ahh     = load_ahh(PATH_AHH, TAHUN)
    df_rls     = load_rls(PATH_RLS, TAHUN)
    df_miskin  = load_kemiskinan(PATH_KEMISKINAN, TAHUN)
    df_keluhan = load_keluhan_kesehatan(PATH_KESEHATAN, TAHUN)
    
    # Gabungkan
    master = gabungkan_master(df_ahh, df_rls, df_miskin, df_keluhan)
    
    # Validasi
    valid = validasi_dataset(master)
    
    # Simpan output
    output_path_csv   = os.path.join(OUTPUT_DIR, "master_dataset.csv")
    output_path_excel = os.path.join(OUTPUT_DIR, "master_dataset.xlsx")
    
    master.to_csv(output_path_csv, index=False, encoding="utf-8-sig")
    master.to_excel(output_path_excel, index=False)
    
    print("\n" + "="*60)
    print("HASIL AKHIR")
    print("="*60)
    print(f"  Master dataset berhasil dibuat!")
    print(f"  Total baris   : {len(master)}")
    print(f"  Total kolom   : {len(master.columns)}")
    print(f"  Kolom         : {list(master.columns)}")
    print(f"\n  File disimpan di:")
    print(f"    → {output_path_csv}")
    print(f"    → {output_path_excel}")
    
    if valid:
        print("\n  ✓ Data VALID dan SIAP untuk diproses PCA + K-Means!")
    else:
        print("\n  [PERHATIAN] Ada isu kualitas data, periksa laporan di atas.")
    
    print("\n  Preview 5 baris pertama:")
    print(master.head().to_string(index=False))
    
    return master


if __name__ == "__main__":
    master = main()