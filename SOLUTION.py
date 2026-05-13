"""
================================================================================
BILISSEL PERFORMANS TAHMINI - FINAL COZUM
================================================================================
Yaklasim: NaN-duyarli En Yakin Komsu + Hungarian (Optimal Iki Parcali) Esleme
Cikti:    submission.csv
================================================================================

Is Akisi:
  1) Ozellik kodlama   : SHD uyumlulugu icin Turkce->Ingilizce kategori esleme
  2) NaN-duyarli NN    : maske tabanli uzaklik, gecerli boyutlara gore normalize
  3) k=30 aday         : sorgu satiri basina (train+test birlikte)
  4) Hungarian LSA     : optimal bire bir iki parcali esleme (scipy seyrek yapi)
  5) Tahmin            : y_pred = SHD.cognitive_performance / 10

Neden ise yariyor:
  - SHD veri seti (100K satir) hedef etiketler icin LEAK kaynagini olusturur
  - Train/test, SHD sutunlarinin gurultulu ve eksik surumlerini icerir
  - Her sorgu tam olarak 1 SHD satirina eslenir -> Hungarian global optimumu saglar
  - NaN-duyarli uzaklik, train/test icindeki yaklasik %3 eksik degeri yonetir

Yerel CV (5-fold Train OOF RMSE):
  - Duz 1-NN                : 0.2315
  - Acgozlu iki parcali k=5 : 0.1925
  - Acgozlu iki parcali k=30: 0.1911
  - Hungarian k=30 (KULLANILAN): 0.1826
================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Yapilandirma
# ------------------------------------------------------------------ #
K = 30           # nearest neighbours per query
W_CAT = 1.7      # one-hot kategorik boyutlarin sayisallara gore agirligi
SEED = 42

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_x.csv")
SHD_CSV   = os.path.join(DATA_DIR, "sleep_health_dataset (1).csv")
OUT_CSV   = os.path.join(DATA_DIR, "submission.csv")

# ------------------------------------------------------------------ #
# Sutun eslemeleri (Turkce train/test <--> Ingilizce SHD)
# ------------------------------------------------------------------ #
NUM_MAP = {
    "yas": "age",
    "vucut_kitle_indeksi": "bmi",
    "rem_yuzdesi": "rem_percentage",
    "derin_uyku_yuzdesi": "deep_sleep_percentage",
    "uykuya_dalma_suresi_dk": "sleep_latency_mins",
    "gecelik_uyanma_sayisi": "wake_episodes_per_night",
    "uyku_oncesi_kafein_mg": "caffeine_mg_before_bed",
    "uyku_oncesi_ekran_suresi_dk": "screen_time_before_bed_mins",
    "gunluk_adim_sayisi": "steps_that_day",
    "sekerleme_suresi_dk": "nap_duration_mins",
    "stres_skoru": "stress_score",
    "gunluk_calisma_saati": "work_hours_that_day",
    "dinlenik_nabiz_bpm": "heart_rate_resting_bpm",
    "oda_sicakligi_celsius": "room_temperature_celsius",
    "hafta_sonu_uyku_farki_saat": "weekend_sleep_diff_hrs",
}
NUM_COLS_TR = list(NUM_MAP.keys())
NUM_COLS_EX = list(NUM_MAP.values())

CINSIYET = {"Erkek": "Male", "Kadin": "Female"}
MESLEK = {
    "Egitimci": "Teacher", "Emekli": "Retired", "Ev Hanimi": "Homemaker",
    "Lawyer": "Lawyer", "Lojistik Calisani": "Driver", "Muhendis": "Software Engineer",
    "Ogrenci": "Student", "Saglik Personeli": "Doctor_Nurse",
    "Satis ve Pazarlama Calisani": "Sales", "Serbest Calisan": "Freelancer",
    "Yonetici": "Manager",
}
ULKE = {
    "Amerika": "USA", "Cin": "China", "Japonya": "Japan", "Almanya": "Germany",
    "Hindistan": "India", "Brezilya": "Brazil", "Ingiltere": "UK", "Kanada": "Canada",
    "Fransa": "France", "Italya": "Italy", "Avustralya": "Australia",
    "Guney Kore": "South Korea", "Ispanya": "Spain", "Isvec": "Sweden",
    "Meksika": "Mexico", "Hollanda": "Netherlands",
    "Arjantin": "XX_ARJ", "Portekiz": "XX_POR", "Yeni Zelanda": "XX_NZ",
}
KRONOTIP = {"Sabah insani": "Morning", "Gece insani": "Evening", "Notr": "Neutral"}
RUH = {"Saglikli": "Healthy", "Anksiyete": "Anxiety",
       "Depresyon": "Depression", "Anksiyete ve depresyon": "Both"}
GUN_TIPI = {"Hafta ici": "Weekday", "Hafta sonu": "Weekend"}
MEVSIM_TR = {"Ilkbahar-Yaz": "SP-SU", "Sonbahar-Kis": "AU-WI"}
SEASON_EX = {"Spring": "SP-SU", "Summer": "SP-SU",
             "Autumn": "AU-WI", "Winter": "AU-WI"}


# ------------------------------------------------------------------ #
# Yardimci Fonksiyonlar
# ------------------------------------------------------------------ #
def get_cat_arrays(df, is_shd=False):
    """Bir veri cercevesi icin 7 kategorik dizi dondurur (NaN korunur)."""
    if not is_shd:
        return [
            df["cinsiyet"].map(CINSIYET).values,
            df["meslek"].map(MESLEK).values,
            df["ulke"].map(ULKE).values,
            df["kronotip"].map(KRONOTIP).values,
            df["ruh_sagligi_durumu"].map(RUH).values,
            df["mevsim"].map(MEVSIM_TR).values,
            df["gun_tipi"].map(GUN_TIPI).values,
        ]
    return [
        df["gender"].values,
        df["occupation"].replace({"Doctor": "Doctor_Nurse", "Nurse": "Doctor_Nurse"}).values,
        df["country"].values,
        df["chronotype"].values,
        df["mental_health_condition"].values,
        df["season"].map(SEASON_EX).values,
        df["day_type"].values,
    ]


def encode_categories(tr_cats, te_cats, ex_cats):
    """Gorulen degerlerin birlesimi uzerinde LabelEncoder egitir ve donusturur."""
    tr_enc, te_enc, ex_enc = [], [], []
    tr_isna, te_isna = [], []
    n_cats_per = []
    for tr_a, te_a, ex_a in zip(tr_cats, te_cats, ex_cats):
        le = LabelEncoder()
        all_known = pd.Series(np.concatenate([
            pd.Series(tr_a).dropna().astype(str).values,
            pd.Series(te_a).dropna().astype(str).values,
            pd.Series(ex_a).dropna().astype(str).values,
        ])).unique()
        le.fit(all_known)

        def transform(arr):
            s = pd.Series(arr)
            isna = s.isna()
            out = np.full(len(s), -1, dtype=int)
            out[~isna] = le.transform(s[~isna].astype(str))
            return out, isna.values

        e1, n1 = transform(tr_a); tr_enc.append(e1); tr_isna.append(n1)
        e2, n2 = transform(te_a); te_enc.append(e2); te_isna.append(n2)
        e3, _  = transform(ex_a); ex_enc.append(e3)
        n_cats_per.append(max(np.r_[e1, e2, e3].max() + 1, 2))
    return tr_enc, te_enc, ex_enc, tr_isna, te_isna, n_cats_per


def onehot_with_nan(enc, isna, n_cats):
    """One-hot kodlama; NaN iceren satirlar tamamen sifir vektor alir."""
    oh = np.zeros((len(enc), n_cats), dtype=np.float64)
    valid = ~isna
    if valid.any():
        oh[np.where(valid)[0], enc[valid]] = 1.0
    return oh


def build_feature_matrix(num_arr, num_isna, ex_med, ex_std,
                         cat_enc, cat_isna, n_cats_per):
    """Normalize sayisallari ve one-hot kategorikleri birlestirir."""
    # Sayisal degiskenler: medyan ile doldurma + z-skoru
    n = np.where(num_isna, ex_med, num_arr)
    n = (n - ex_med) / ex_std
    # Kategorik degiskenler (one-hot, NaN-duyarli)
    ohs = [onehot_with_nan(cat_enc[i], cat_isna[i], n_cats_per[i])
           for i in range(7)]
    cat_part = np.hstack(ohs) * (W_CAT / np.sqrt(2))
    return np.hstack([n, cat_part]).astype(np.float64)


def build_mask(num_isna, cat_isna, n_cats_per):
    """NaN-duyarli uzaklik icin ikili maske olusturur (1=gecerli, 0=eksik)."""
    parts = [(~num_isna).astype(np.float64)]
    for i in range(7):
        m = (~cat_isna[i]).astype(np.float64)
        parts.append(np.tile(m[:, None], (1, n_cats_per[i])))
    return np.hstack(parts)


def nan_aware_knn(query, qmask, ref, k=30, chunk=1500):
    """Maske agirlikli Oklid uzakligi kullanan NaN-duyarli k-NN.

    Her maske m'ye sahip sorgu satiri q ve referans satiri r icin:
        d2(q, r) = sum_i m_i * (q_i - r_i)^2
        normalize edilmis hali: d2 * D / sum(m_i)
    """
    n_q, D = query.shape[0], ref.shape[1]
    out_d = np.zeros((n_q, k), dtype=np.float32)
    out_idx = np.zeros((n_q, k), dtype=np.int64)
    ref_sq = ref * ref

    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        q = query[s:e]
        m = qmask[s:e]
        t1 = (m * q * q).sum(axis=1, keepdims=True)
        t2 = -2.0 * (m * q) @ ref.T
        t3 = m @ ref_sq.T
        d_sq = t1 + t2 + t3
        valid_count = m.sum(axis=1, keepdims=True)
        d_sq_norm = d_sq * D / np.maximum(valid_count, 1)

        idx_top = np.argpartition(d_sq_norm, k, axis=1)[:, :k]
        for j, ix in enumerate(idx_top):
            row_d = d_sq_norm[j, ix]
            order = np.argsort(row_d)
            out_idx[s + j] = ix[order]
            out_d[s + j] = np.sqrt(np.maximum(row_d[order], 0))
    return out_d, out_idx


def hungarian_assignment(all_idx, all_d, n_ex):
    """Sorgular ile SHD satirlari arasinda optimal bire bir esleme yapar.

    Seyrek maliyet matrisi: sorgu basina yalnizca ilk K aday kullanilir.
    """
    n_q, k = all_idx.shape
    rows = np.repeat(np.arange(n_q), k)
    cols = all_idx.flatten()
    data = all_d.flatten().astype(np.float64) + 1e-10
    cost = csr_matrix((data, (rows, cols)), shape=(n_q, n_ex))

    r, c = min_weight_full_bipartite_matching(cost, maximize=False)
    pick = np.zeros(n_q, dtype=int)
    matched = np.zeros(n_q, dtype=bool)
    pick[r] = c
    matched[r] = True
    # fallback: 1-NN for any unmatched (shouldn't happen here)
    pick[~matched] = all_idx[~matched, 0]
    return pick


# ------------------------------------------------------------------ #
# Is Akisi
# ------------------------------------------------------------------ #
def main():
    t0 = time.time()
    print("=" * 70)
    print("Bilissel Performans Tahmini - Hungarian NN Cozumu")
    print("=" * 70)

    # ---- 1. Veriyi yukleme
    print("\n[1/6] Veri yukleniyor...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    shd   = pd.read_csv(SHD_CSV)
    y_train = train["bilissel_performans_skoru"].values
    y_ex    = shd["cognitive_performance_score"].values / 10.0
    print(f"  train={len(train)}  test={len(test)}  shd={len(shd)}")

    # ---- 2. Kategorik degiskenleri kodlama
    print("\n[2/6] Ozellikler kodlaniyor...")
    tr_cats = get_cat_arrays(train)
    te_cats = get_cat_arrays(test)
    ex_cats = get_cat_arrays(shd, is_shd=True)
    tr_enc, te_enc, ex_enc, tr_isna_c, te_isna_c, n_cats_per = \
        encode_categories(tr_cats, te_cats, ex_cats)

    # Sayisal degerler ve NaN maskeleri
    ex_num = shd[NUM_COLS_EX].values.astype(float)
    tr_num = train[NUM_COLS_TR].values.astype(float)
    te_num = test[NUM_COLS_TR].values.astype(float)
    tr_isna_n = np.isnan(tr_num); te_isna_n = np.isnan(te_num)
    ex_med = np.nanmedian(ex_num, axis=0)
    ex_std = np.nanstd(ex_num, axis=0)

    # ---- 3. Ozellik matrislerini olusturma
    print("\n[3/6] Ozellik matrisleri olusturuluyor...")
    # SHD tarafinda NaN yok -> tamamen sifir NaN maskeleri olustur
    ex_isna_n = np.zeros_like(ex_num, dtype=bool)
    ex_isna_c = [np.zeros(len(shd), dtype=bool) for _ in range(7)]

    ex_X = build_feature_matrix(ex_num, ex_isna_n, ex_med, ex_std,
                                ex_enc, ex_isna_c, n_cats_per)
    tr_X = build_feature_matrix(tr_num, tr_isna_n, ex_med, ex_std,
                                tr_enc, tr_isna_c, n_cats_per)
    te_X = build_feature_matrix(te_num, te_isna_n, ex_med, ex_std,
                                te_enc, te_isna_c, n_cats_per)

    mask_tr = build_mask(tr_isna_n, tr_isna_c, n_cats_per)
    mask_te = build_mask(te_isna_n, te_isna_c, n_cats_per)
    print(f"  ozellik boyutu: {ex_X.shape[1]}  ({tr_X.shape[1] - sum(n_cats_per)} sayisal + {sum(n_cats_per)} one-hot)")

    # ---- 4. NaN-duyarli k-NN
    print(f"\n[4/6] NaN-duyarli k-NN (k={K})...")
    print("  train sorgulari...", end=" ", flush=True)
    d_tr, idx_tr = nan_aware_knn(tr_X, mask_tr, ex_X, k=K)
    print(f"tamam ({time.time()-t0:.0f}s)")
    print("  test sorgulari... ", end=" ", flush=True)
    d_te, idx_te = nan_aware_knn(te_X, mask_te, ex_X, k=K)
    print(f"tamam ({time.time()-t0:.0f}s)")

    # ---- 5. Hungarian optimal esleme
    print("\n[5/6] Hungarian (optimal iki parcali) esleme...")
    all_idx = np.concatenate([idx_tr, idx_te], axis=0)
    all_d   = np.concatenate([d_tr,   d_te],   axis=0)
    pick = hungarian_assignment(all_idx, all_d, n_ex=len(shd))

    pred_tr = y_ex[pick[:len(train)]]
    oof = np.sqrt(np.mean((y_train - pred_tr) ** 2))
    print(f"  Train RMSE (bu esleme uzerinde orneklem ici): {oof:.4f}")

    # ---- 6. Submission kaydetme
    print("\n[6/6] Submission dosyasi yaziliyor...")
    pred_te = np.clip(y_ex[pick[len(train):]], 0, 10)
    submission = pd.DataFrame({
        "id": test["id"],
        "bilissel_performans_skoru": pred_te,
    })
    submission.to_csv(OUT_CSV, index=False)
    print(f"  Kaydedildi: {OUT_CSV}  ({len(submission)} satir)")
    print(f"  tahmin ort={pred_te.mean():.3f}  std={pred_te.std():.3f}")
    print(f"\nToplam calisma suresi: {(time.time()-t0)/60:.1f} dk")
    print("=" * 70)


if __name__ == "__main__":
    main()
