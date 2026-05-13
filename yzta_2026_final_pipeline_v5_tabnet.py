"""
================================================================================
YZTA 2026 DATATHON - BILISSEL PERFORMANS SKORU TAHMINI
Final Pipeline (v5 + TabNet)



YAKLASIM OZETI:
  1. Veri temizleme (Turkce/Ingilizce karisik ulke ve meslek isimleri normalize edilir)
  2. Feature engineering (uyku/stres/aktivite turevleri, 27 yeni ozellik)
  3. Hedef Ortalamasi Encoding (KFold-safe, smoothing alpha=5)
  4. Group statistics (meslek, ulke, ruh_sagligi bazinda mean/std)
  5. 5-Fold CV ile 9 farkli model:
     - LightGBM (multi-seed)         : 3 seed averaging
     - CatBoost                      : kategorik dostu
     - LightGBM + Pseudo-Labeling    : test setini soft labels ile training'e ekler
     - CatBoost + Pseudo-Labeling
     - LightGBM Round 2 (Self-distill): yeni tahminlerle yeniden PL
     - CatBoost Round 2 (Self-distill)
     - Tabular MLP                   : PyTorch, embedding'li
     - Tabular MLP (multi-seed)
     - Tabular MLP + Pseudo-Labeling
  6. Forward selection ile optimal ensemble agirliklari
  7. Polynomial-5 bias correction (KFold-safe)
  8. v5 ensemble + TabNet mikro blend (96.5% / 3.5%)

ON KOSUL:
  - Adversarial validation AUC = 0.509 (train-test ayni dagilimda) -> PL guvenli


================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
N_SPLITS = 5
TARGET = 'bilissel_performans_skoru'

# Kaggle path - kendi dataset adiniza gore degistirin
INPUT_DIR  = '/kaggle/input/yzta-2026-datathon'
OUTPUT_DIR = '/kaggle/working'

# Lokal kullanim icin alternatif:
# INPUT_DIR = '.'
# OUTPUT_DIR = '.'

TRAIN_PATH = os.path.join(INPUT_DIR, 'train.csv')
TEST_PATH  = os.path.join(INPUT_DIR, 'test_x.csv')

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("YZTA 2026 DATATHON - FINAL PIPELINE (v5 + TabNet)")
print("=" * 80)
T_START = time.time()


# ============================================================================
# 1) VERI YUKLEME
# ============================================================================
print("\n[1/9] Veri yukleniyor...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
y = train[TARGET].values
test_ids = test['id'].values
print(f"  Train: {train.shape}, Test: {test.shape}")


# ============================================================================
# 2) VERI TEMIZLEME
# Tespit: ulke ve meslek sutunlarinda Turkce/Ingilizce karisik etiketler var.
# Bu iki dilli karisikligi normalize ederek kategorik kardinaliteyi dusurmek
# kritik bir veri kalitesi iyileştirmesidir.
# ============================================================================
print("\n[2/9] Veri temizleniyor (cift dilli etiket normalizasyonu)...")
ulke_map = {
    'South Korea': 'Guney Kore',
    'Spain':       'Ispanya',
    'Sweden':      'Isvec',
    'Mexico':      'Meksika',
    'Netherlands': 'Hollanda',
}
meslek_map = {'Lawyer': 'Avukat'}

for df in (train, test):
    df['ulke']   = df['ulke'].replace(ulke_map)
    df['meslek'] = df['meslek'].replace(meslek_map)


# ============================================================================
# 3) FEATURE ENGINEERING
# Alan bilgisine dayali 27 yeni ozellik:
#   - Uyku mimarisi (REM/derin oranlari, etkin uyku)
#   - Uyku bozulma kompozit endeksi
#   - Stres etkilesimleri (en guclu sinyal -0.58 korr)
#   - Aktivite skorlari
#   - Demografik kategorizasyon
#   - Cevresel konfor (oda sicakligi)
# ============================================================================
print("\n[3/9] Feature engineering...")

def feature_engineer(df):
    df = df.copy()
    # Uyku mimarisi
    df['toplam_uyku_evre_yuzdesi'] = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']
    df['hafif_uyku_yuzdesi']       = 100 - df['toplam_uyku_evre_yuzdesi']
    df['rem_derin_orani']          = df['rem_yuzdesi'] / (df['derin_uyku_yuzdesi'] + 1e-3)
    df['rem_derin_fark']           = df['rem_yuzdesi'] - df['derin_uyku_yuzdesi']
    df['etkin_uyku_yuzde']         = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi'] - df['gecelik_uyanma_sayisi']
    # Uyku bozulma kompozit endeksi
    df['uyku_bozulma'] = (df['uykuya_dalma_suresi_dk']
                          + df['gecelik_uyanma_sayisi'] * 10
                          + df['uyku_oncesi_ekran_suresi_dk'] / 10)
    df['uyku_kalitesi'] = (df['derin_uyku_yuzdesi']
                            - df['gecelik_uyanma_sayisi'] * 2
                            - df['uykuya_dalma_suresi_dk'] / 10)
    df['kafein_ekran'] = df['uyku_oncesi_kafein_mg'] * df['uyku_oncesi_ekran_suresi_dk']
    df['uyanma_dalma'] = df['gecelik_uyanma_sayisi'] * df['uykuya_dalma_suresi_dk']
    # Stres etkilesimleri (baskin sinyalin turevleri)
    df['stres_uyku']    = df['stres_skoru'] * df['uyku_bozulma']
    df['stres_calisma'] = df['stres_skoru'] * df['gunluk_calisma_saati']
    df['stres_kafein']  = df['stres_skoru'] * df['uyku_oncesi_kafein_mg']
    df['stres_nabiz']   = df['stres_skoru'] * df['dinlenik_nabiz_bpm']
    df['stres_yas']     = df['stres_skoru'] * df['yas']
    df['stres_kare']    = df['stres_skoru'] ** 2
    df['stres_log']     = np.log1p(df['stres_skoru'].clip(lower=0))
    df['calisma_yogunluk'] = df['gunluk_calisma_saati'] * df['stres_skoru']
    df['hafta_sonu_etki']  = df['hafta_sonu_uyku_farki_saat'] * df['stres_skoru']
    # Aktivite
    df['adim_calisma_orani'] = df['gunluk_adim_sayisi'] / (df['gunluk_calisma_saati'] + 1e-3)
    df['adim_yas']       = df['gunluk_adim_sayisi'] / (df['yas'] + 1)
    df['aktivite_skoru'] = df['gunluk_adim_sayisi'] / 1000 + df['sekerleme_suresi_dk'] / 30
    # Demografik kategorizasyon
    df['yas_grubu'] = pd.cut(df['yas'], bins=[0, 25, 35, 45, 55, 100], labels=False).astype(float)
    df['bmi_grubu'] = pd.cut(df['vucut_kitle_indeksi'], bins=[0, 18.5, 25, 30, 100], labels=False).astype(float)
    df['yas_bmi']   = df['yas'] * df['vucut_kitle_indeksi']
    df['nabiz_yas'] = df['dinlenik_nabiz_bpm'] / (df['yas'] + 1)
    df['nabiz_bmi'] = df['dinlenik_nabiz_bpm'] / (df['vucut_kitle_indeksi'] + 1e-3)
    # Cevresel konfor
    df['sicaklik_konfor'] = np.abs(df['oda_sicakligi_celsius'] - 21)
    df['sicaklik_kare']   = (df['oda_sicakligi_celsius'] - 21) ** 2
    return df

train = feature_engineer(train)
test  = feature_engineer(test)
print(f"  FE sonrasi: {train.shape[1]} sutun")

cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']


# ============================================================================
# 4) MODEL 1: LightGBM (Multi-seed, 3 farkli seed averaging)
# ============================================================================
print("\n[4/9] Model 1: LightGBM (multi-seed)...")

train_lgb = train.copy(); test_lgb = test.copy()
for col in cat_cols:
    train_lgb[col] = train_lgb[col].astype('category')
    test_lgb[col]  = pd.Categorical(test_lgb[col], categories=train_lgb[col].cat.categories)

features_lgb = [c for c in train_lgb.columns if c not in ['id', TARGET]]
X_lgb = train_lgb[features_lgb]
X_test_lgb = test_lgb[features_lgb]

lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.025, 'num_leaves': 95, 'max_depth': -1,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.80, 'bagging_fraction': 0.85, 'bagging_freq': 5,
    'lambda_l1': 0.2, 'lambda_l2': 0.2, 'min_gain_to_split': 0.02,
    'verbose': -1, 'seed': SEED, 'n_jobs': -1,
}

oof_lgb  = np.zeros(len(train))
pred_lgb = np.zeros(len(test))
for seed in [42, 2024, 7]:
    params_s = {**lgb_params, 'seed': seed}
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_s  = np.zeros(len(train))
    pred_s = np.zeros(len(test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_lgb)):
        ds_tr = lgb.Dataset(X_lgb.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
        ds_va = lgb.Dataset(X_lgb.iloc[val_idx], y[val_idx], categorical_feature=cat_cols, reference=ds_tr)
        m = lgb.train(params_s, ds_tr, num_boost_round=8000, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)])
        oof_s[val_idx]  = m.predict(X_lgb.iloc[val_idx], num_iteration=m.best_iteration)
        pred_s         += m.predict(X_test_lgb, num_iteration=m.best_iteration) / N_SPLITS
    print(f"  seed={seed}: OOF RMSE = {np.sqrt(mean_squared_error(y, oof_s)):.5f}")
    oof_lgb  += oof_s  / 3
    pred_lgb += pred_s / 3
print(f"  LightGBM Multi-seed OOF RMSE: {np.sqrt(mean_squared_error(y, oof_lgb)):.5f}")


# ============================================================================
# 5) MODEL 2: CatBoost (kategorik dostu, beklenen en iyi tek model)
# ============================================================================
print("\n[5/9] Model 2: CatBoost...")

train_cb = train.copy(); test_cb = test.copy()
for col in cat_cols:
    train_cb[col] = train_cb[col].astype(str).fillna('missing').replace('nan', 'missing')
    test_cb[col]  = test_cb[col].astype(str).fillna('missing').replace('nan', 'missing')

features_cb = [c for c in train_cb.columns if c not in ['id', TARGET]]
X_cb = train_cb[features_cb]
X_test_cb = test_cb[features_cb]
cat_idx_cb = [features_cb.index(c) for c in cat_cols]

oof_cb  = np.zeros(len(train))
pred_cb = np.zeros(len(test))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb)):
    m = CatBoostRegressor(
        iterations=2500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', random_seed=SEED,
        early_stopping_rounds=150, verbose=False, thread_count=-1,
    )
    m.fit(Pool(X_cb.iloc[tr_idx], y[tr_idx], cat_features=cat_idx_cb),
          eval_set=Pool(X_cb.iloc[val_idx], y[val_idx], cat_features=cat_idx_cb),
          use_best_model=True)
    oof_cb[val_idx] = m.predict(X_cb.iloc[val_idx])
    pred_cb        += m.predict(X_test_cb) / N_SPLITS
    print(f"  Fold {fold+1}: RMSE = {np.sqrt(mean_squared_error(y[val_idx], oof_cb[val_idx])):.5f}")
print(f"  CatBoost OOF RMSE: {np.sqrt(mean_squared_error(y, oof_cb)):.5f}")


# ============================================================================
# 6) MODELS 3-6: Pseudo-Labeling (Round 1 + Round 2 self-distillation)
# Adversarial AUC=0.51 olarak gosterilmistir -> PL guvenli
# ============================================================================
print("\n[6/9] Modeller 3-6: Pseudo-Labeling + Self-Distillation...")

def blend_optimize(oof_list, y_arr):
    """Forward selection icin yardimci: optimal weighted blend."""
    n = len(oof_list)
    def loss(w):
        w = np.abs(w); w = w / w.sum()
        return np.sqrt(mean_squared_error(y_arr, sum(wi*o for wi, o in zip(w, oof_list))))
    res = minimize(loss, x0=np.ones(n)/n, method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 3000})
    w = np.abs(res.x); w = w / w.sum()
    return w

# Round 1 PL: ilk LGBM+CB blend tahmini
w_r1 = blend_optimize([oof_lgb, oof_cb], y)
test_pseudo_r1 = w_r1[0] * pred_lgb + w_r1[1] * pred_cb
test_pseudo_r1 = np.clip(test_pseudo_r1, 0, 10)

# ----- LightGBM + PL Round 1 -----
print("  LightGBM + Pseudo-Labels (Round 1)...")
test_pl_lgb = test_lgb.copy(); test_pl_lgb[TARGET] = test_pseudo_r1
X_pl_lgb = test_pl_lgb[features_lgb]
oof_lgb_pl  = np.zeros(len(train))
pred_lgb_pl = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_lgb)):
    X_tr = pd.concat([X_lgb.iloc[tr_idx], X_pl_lgb], ignore_index=True)
    y_tr = np.concatenate([y[tr_idx], test_pseudo_r1])
    ds_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    ds_va = lgb.Dataset(X_lgb.iloc[val_idx], y[val_idx], categorical_feature=cat_cols, reference=ds_tr)
    m = lgb.train(lgb_params, ds_tr, num_boost_round=8000, valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)])
    oof_lgb_pl[val_idx] = m.predict(X_lgb.iloc[val_idx], num_iteration=m.best_iteration)
    pred_lgb_pl        += m.predict(X_test_lgb, num_iteration=m.best_iteration) / N_SPLITS
print(f"  LGBM-PL OOF RMSE: {np.sqrt(mean_squared_error(y, oof_lgb_pl)):.5f}")

# ----- CatBoost + PL Round 1 -----
print("  CatBoost + Pseudo-Labels (Round 1)...")
test_pl_cb = test_cb.copy(); test_pl_cb[TARGET] = test_pseudo_r1
X_pl_cb = test_pl_cb[features_cb]
oof_cb_pl  = np.zeros(len(train))
pred_cb_pl = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb)):
    X_tr = pd.concat([X_cb.iloc[tr_idx], X_pl_cb], ignore_index=True)
    y_tr = np.concatenate([y[tr_idx], test_pseudo_r1])
    m = CatBoostRegressor(
        iterations=1000, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', random_seed=SEED,
        early_stopping_rounds=80, verbose=False, thread_count=-1,
    )
    m.fit(Pool(X_tr, y_tr, cat_features=cat_idx_cb),
          eval_set=Pool(X_cb.iloc[val_idx], y[val_idx], cat_features=cat_idx_cb),
          use_best_model=True)
    oof_cb_pl[val_idx] = m.predict(X_cb.iloc[val_idx])
    pred_cb_pl        += m.predict(X_test_cb) / N_SPLITS
print(f"  CB-PL OOF RMSE: {np.sqrt(mean_squared_error(y, oof_cb_pl)):.5f}")

# ----- Round 2: Self-Distillation -----
print("  Round 2 Self-Distillation...")
oof_r1_list  = [oof_lgb, oof_cb, oof_lgb_pl, oof_cb_pl]
pred_r1_list = [pred_lgb, pred_cb, pred_lgb_pl, pred_cb_pl]
w_r2 = blend_optimize(oof_r1_list, y)
test_pseudo_r2 = sum(wi * p for wi, p in zip(w_r2, pred_r1_list))
test_pseudo_r2 = np.clip(test_pseudo_r2, 0, 10)

# LightGBM Round 2
test_pl_lgb[TARGET] = test_pseudo_r2
X_pl_lgb = test_pl_lgb[features_lgb]
oof_lgb_r2  = np.zeros(len(train))
pred_lgb_r2 = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_lgb)):
    X_tr = pd.concat([X_lgb.iloc[tr_idx], X_pl_lgb], ignore_index=True)
    y_tr = np.concatenate([y[tr_idx], test_pseudo_r2])
    ds_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    ds_va = lgb.Dataset(X_lgb.iloc[val_idx], y[val_idx], categorical_feature=cat_cols, reference=ds_tr)
    m = lgb.train(lgb_params, ds_tr, num_boost_round=8000, valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)])
    oof_lgb_r2[val_idx] = m.predict(X_lgb.iloc[val_idx], num_iteration=m.best_iteration)
    pred_lgb_r2        += m.predict(X_test_lgb, num_iteration=m.best_iteration) / N_SPLITS
print(f"  LGBM-R2 OOF RMSE: {np.sqrt(mean_squared_error(y, oof_lgb_r2)):.5f}")

# CatBoost Round 2
test_pl_cb[TARGET] = test_pseudo_r2
X_pl_cb = test_pl_cb[features_cb]
oof_cb_r2  = np.zeros(len(train))
pred_cb_r2 = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb)):
    X_tr = pd.concat([X_cb.iloc[tr_idx], X_pl_cb], ignore_index=True)
    y_tr = np.concatenate([y[tr_idx], test_pseudo_r2])
    m = CatBoostRegressor(
        iterations=1000, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', random_seed=SEED,
        early_stopping_rounds=80, verbose=False, thread_count=-1,
    )
    m.fit(Pool(X_tr, y_tr, cat_features=cat_idx_cb),
          eval_set=Pool(X_cb.iloc[val_idx], y[val_idx], cat_features=cat_idx_cb),
          use_best_model=True)
    oof_cb_r2[val_idx] = m.predict(X_cb.iloc[val_idx])
    pred_cb_r2        += m.predict(X_test_cb) / N_SPLITS
print(f"  CB-R2 OOF RMSE: {np.sqrt(mean_squared_error(y, oof_cb_r2)):.5f}")


# ============================================================================
# 7) MODELS 7-9: Tabular MLP (PyTorch) - GBDT olmayan diversity
# ============================================================================
print("\n[7/9] Modeller 7-9: Tabular MLP (PyTorch)...")

# MLP icin ek FE: KFold target encoding (alpha=5) + group statistics
ALPHA_TE = 5
gm = float(y.mean())

train_mlp = train.copy(); test_mlp = test.copy()

# Target encoding (KFold-safe icin train icin OOF, test icin tum train uzerinden)
for col in cat_cols:
    g = train_mlp.groupby(col, observed=True)[TARGET].agg(['sum', 'count'])
    te_map = ((g['sum'] + ALPHA_TE * gm) / (g['count'] + ALPHA_TE)).to_dict()
    test_mlp[f'te_{col}']  = test_mlp[col].map(te_map).fillna(gm).astype(np.float64)
    train_mlp[f'te_{col}'] = np.float64(gm)

kf_te = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for tr_i, va_i in kf_te.split(train_mlp):
    sub_tr = train_mlp.iloc[tr_i]
    for col in cat_cols:
        g = sub_tr.groupby(col, observed=True)[TARGET].agg(['sum', 'count'])
        te_map = ((g['sum'] + ALPHA_TE * gm) / (g['count'] + ALPHA_TE)).to_dict()
        train_mlp.iloc[va_i, train_mlp.columns.get_loc(f'te_{col}')] = (
            train_mlp.iloc[va_i][col].map(te_map).fillna(gm).values
        )

# Group statistics
train_mlp[TARGET] = y
combined = pd.concat([train_mlp, test_mlp], axis=0, ignore_index=True)
key_groupbys = ['meslek', 'ulke', 'ruh_sagligi_durumu', 'kronotip']
key_metrics  = ['stres_skoru', 'gunluk_calisma_saati', 'uyku_bozulma',
                'rem_yuzdesi', 'derin_uyku_yuzdesi']
for gcol in key_groupbys:
    for mcol in key_metrics:
        for stat in ['mean', 'std']:
            combined[f'{gcol}_{mcol}_{stat}'] = combined.groupby(gcol, observed=True)[mcol].transform(stat)

n_train = len(train_mlp)
train_mlp = combined.iloc[:n_train].reset_index(drop=True)
test_mlp  = combined.iloc[n_train:].reset_index(drop=True).drop(columns=[TARGET])
train_mlp[TARGET] = y

# Categorical -> integer encoding
cat_sizes = {}
for col in cat_cols:
    vals = pd.concat([train_mlp[col], test_mlp[col]], ignore_index=True).astype(str).fillna('missing').replace('nan', 'missing')
    cats = sorted(vals.unique())
    cat_sizes[col] = len(cats) + 1
    cat_to_int = {c: i + 1 for i, c in enumerate(cats)}
    train_mlp[col] = train_mlp[col].astype(str).fillna('missing').replace('nan', 'missing').map(cat_to_int).fillna(0).astype(int)
    test_mlp[col]  = test_mlp[col].astype(str).fillna('missing').replace('nan', 'missing').map(cat_to_int).fillna(0).astype(int)

num_cols_mlp = [c for c in train_mlp.columns if c not in ['id', TARGET] + cat_cols]
for c in num_cols_mlp:
    med = train_mlp[c].median()
    train_mlp[c] = train_mlp[c].fillna(med).replace([np.inf, -np.inf], med)
    test_mlp[c]  = test_mlp[c].fillna(med).replace([np.inf, -np.inf], med)

scaler = StandardScaler()
train_num = scaler.fit_transform(train_mlp[num_cols_mlp].values).astype(np.float32)
test_num  = scaler.transform(test_mlp[num_cols_mlp].values).astype(np.float32)
train_cat = train_mlp[cat_cols].values.astype(np.int64)
test_cat  = test_mlp[cat_cols].values.astype(np.int64)
y_f32 = y.astype(np.float32)

print(f"  MLP girdi: {train_num.shape[1]} numeric + {len(cat_cols)} categorical")

class TabularMLP(nn.Module):
    """Embedding'li MLP: kategorik degiskenler embedding tablolarinda saklanir,
    sayisal degiskenler BatchNorm ile normalize edilir."""
    def __init__(self, num_size, cat_sizes_dict):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_sizes_dict[c], min(50, (cat_sizes_dict[c] + 1) // 2))
            for c in cat_cols
        ])
        emb_total = sum(min(50, (cat_sizes_dict[c] + 1) // 2) for c in cat_cols)
        self.bn = nn.BatchNorm1d(num_size)
        self.net = nn.Sequential(
            nn.Linear(num_size + emb_total, 256), nn.SiLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.SiLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x_num, x_cat):
        x_num = self.bn(x_num)
        emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat([x_num] + emb, dim=1)).squeeze(-1)

def train_mlp_kfold(seeds, use_pseudo=False, pseudo_labels=None,
                    epochs=20, batch_size=1024, patience=5, lr=2e-3):
    """Multi-seed KFold MLP egitimi (opsiyonel pseudo-labeling)."""
    oof_total  = np.zeros(len(train_mlp))
    pred_total = np.zeros(len(test_mlp))
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        kf_inner = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof  = np.zeros(len(train_mlp))
        pred = np.zeros(len(test_mlp))
        for fold, (tr_idx, val_idx) in enumerate(kf_inner.split(train_num)):
            if use_pseudo:
                X_tr_n = np.concatenate([train_num[tr_idx], test_num])
                X_tr_c = np.concatenate([train_cat[tr_idx], test_cat])
                y_tr   = np.concatenate([y_f32[tr_idx], pseudo_labels.astype(np.float32)])
            else:
                X_tr_n = train_num[tr_idx]
                X_tr_c = train_cat[tr_idx]
                y_tr   = y_f32[tr_idx]

            Xtn = torch.from_numpy(X_tr_n); Xtc = torch.from_numpy(X_tr_c); yt  = torch.from_numpy(y_tr)
            Xvn = torch.from_numpy(train_num[val_idx]); Xvc = torch.from_numpy(train_cat[val_idx])
            Xen = torch.from_numpy(test_num); Xec = torch.from_numpy(test_cat)
            loader = DataLoader(TensorDataset(Xtn, Xtc, yt), batch_size=batch_size, shuffle=True)

            model = TabularMLP(num_size=train_num.shape[1], cat_sizes_dict=cat_sizes)
            opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            loss_fn = nn.MSELoss()

            best_rmse, best_oof, best_pred, bad = float('inf'), None, None, 0
            for ep in range(epochs):
                model.train()
                for xb_n, xb_c, yb in loader:
                    opt.zero_grad()
                    p = model(xb_n, xb_c)
                    loss = loss_fn(p, yb)
                    loss.backward()
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    vp = model(Xvn, Xvc).numpy()
                r = np.sqrt(mean_squared_error(y[val_idx], vp))
                if r < best_rmse:
                    best_rmse, best_oof = r, vp
                    with torch.no_grad():
                        best_pred = model(Xen, Xec).numpy()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience: break
            oof[val_idx] = best_oof
            pred += best_pred / N_SPLITS

        print(f"    seed {seed}: OOF RMSE = {np.sqrt(mean_squared_error(y, oof)):.5f}")
        oof_total  += oof  / len(seeds)
        pred_total += pred / len(seeds)
    return oof_total, pred_total

# MLP (tek seed)
print("  MLP (tek seed)...")
oof_mlp, pred_mlp = train_mlp_kfold(seeds=[42], epochs=30, batch_size=512)
print(f"  MLP OOF RMSE: {np.sqrt(mean_squared_error(y, oof_mlp)):.5f}")

# MLP Multi-seed (variance reduction)
print("  MLP Multi-seed...")
oof_mlp_ms, pred_mlp_ms = train_mlp_kfold(seeds=[42, 2024], epochs=15)
print(f"  MLP-MS OOF RMSE: {np.sqrt(mean_squared_error(y, oof_mlp_ms)):.5f}")

# MLP + Pseudo-Labels (en yuksek MLP kazanci)
print("  MLP + Pseudo-Labels...")
oof_gbdt_list  = [oof_lgb, oof_cb, oof_lgb_pl, oof_cb_pl, oof_lgb_r2, oof_cb_r2]
pred_gbdt_list = [pred_lgb, pred_cb, pred_lgb_pl, pred_cb_pl, pred_lgb_r2, pred_cb_r2]
w_gbdt = blend_optimize(oof_gbdt_list, y)
test_pseudo_for_mlp = sum(wi * p for wi, p in zip(w_gbdt, pred_gbdt_list))
test_pseudo_for_mlp = np.clip(test_pseudo_for_mlp, 0, 10)
oof_mlp_pl, pred_mlp_pl = train_mlp_kfold(
    seeds=[42, 2024], use_pseudo=True, pseudo_labels=test_pseudo_for_mlp, epochs=15
)
print(f"  MLP-PL OOF RMSE: {np.sqrt(mean_squared_error(y, oof_mlp_pl)):.5f}")


# ============================================================================
# 8) v5 ENSEMBLE (Forward Selection + Bias Correction)
# 9 modelin optimal weighted blend'i, ardindan polynomial-5 bias correction
# ============================================================================
print("\n[8/9] v5 Ensemble (Forward Selection + Bias Correction)...")

oofs = {
    'lgb':    oof_lgb,    'cb':     oof_cb,
    'lgb_pl': oof_lgb_pl, 'cb_pl':  oof_cb_pl,
    'lgb_r2': oof_lgb_r2, 'cb_r2':  oof_cb_r2,
    'mlp':    oof_mlp,    'mlp_ms': oof_mlp_ms, 'mlp_pl': oof_mlp_pl,
}
preds = {
    'lgb':    pred_lgb,    'cb':     pred_cb,
    'lgb_pl': pred_lgb_pl, 'cb_pl':  pred_cb_pl,
    'lgb_r2': pred_lgb_r2, 'cb_r2':  pred_cb_r2,
    'mlp':    pred_mlp,    'mlp_ms': pred_mlp_ms, 'mlp_pl': pred_mlp_pl,
}

print("\n  Bireysel OOF RMSE'ler:")
for k, o in oofs.items():
    print(f"    {k:9s}: {np.sqrt(mean_squared_error(y, o)):.5f}")

# Forward selection: greedy olarak en iyi alt-kume insa et
keys = list(oofs.keys())
best_set, best_rmse = [], float('inf')
final_oof, final_pred, final_w = None, None, None

for k in keys:
    rmse = np.sqrt(mean_squared_error(y, oofs[k]))
    if rmse < best_rmse:
        best_rmse, best_set = rmse, [k]
        final_oof, final_pred, final_w = oofs[k], preds[k], {k: 1.0}

def opt_weights(oof_list, pred_list, y_arr):
    n = len(oof_list)
    def loss(w):
        w = np.abs(w); w = w / w.sum()
        return np.sqrt(mean_squared_error(y_arr, sum(wi * o for wi, o in zip(w, oof_list))))
    res = minimize(loss, x0=np.ones(n)/n, method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 5000})
    w = np.abs(res.x); w = w / w.sum()
    return (w,
            sum(wi * o for wi, o in zip(w, oof_list)),
            sum(wi * p for wi, p in zip(w, pred_list)),
            res.fun)

improved = True
while improved:
    improved = False
    for k in keys:
        if k in best_set: continue
        cand = best_set + [k]
        ol = [oofs[c]  for c in cand]
        pl = [preds[c] for c in cand]
        w, ob, pb, rmse = opt_weights(ol, pl, y)
        if rmse < best_rmse - 1e-5:
            best_rmse, best_set = rmse, cand
            final_oof, final_pred = ob, pb
            final_w = dict(zip(cand, w))
            improved = True
    if improved:
        print(f"\n  Eklendi: {best_set} -> OOF RMSE = {best_rmse:.5f}")

print(f"\n  v5 Ensemble agirliklari:")
for k, w in sorted(final_w.items(), key=lambda x: -x[1]):
    print(f"    {k:9s}: {w:.4f}")

# Polynomial-5 bias correction (KFold-safe)
print("\n  Polynomial-5 bias correction...")
kf_bc = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_bc  = np.zeros(len(y))
pred_bc_list = []
for tr_i, va_i in kf_bc.split(final_oof):
    coefs = np.polyfit(final_oof[tr_i], y[tr_i], deg=5)
    oof_bc[va_i] = np.polyval(coefs, final_oof[va_i])
    pred_bc_list.append(np.polyval(coefs, final_pred))
pred_bc = np.mean(pred_bc_list, axis=0)
oof_bc  = np.clip(oof_bc, 0, 10)
pred_bc = np.clip(pred_bc, 0, 10)

if np.sqrt(mean_squared_error(y, oof_bc)) < np.sqrt(mean_squared_error(y, final_oof)):
    print(f"  Bias correction kullanildi: {np.sqrt(mean_squared_error(y, oof_bc)):.5f}")
    v5_oof, v5_pred = oof_bc, pred_bc
else:
    print(f"  Bias correction kullanilmadi.")
    v5_oof, v5_pred = final_oof, np.clip(final_pred, 0, 10)

print(f"\n  v5 Ensemble OOF RMSE: {np.sqrt(mean_squared_error(y, v5_oof)):.5f}")


# ============================================================================
# 9) MODEL 10: TabNet + Final Mikro Blend
# Dikkat-tabanli tabular model. v5 ile dusuk korelasyon (~0.99) -> diversity
# Final: %96.5 v5 + %3.5 TabNet
# ============================================================================
print("\n[9/9] TabNet + Final Mikro Blend...")

# TabNet icin Label Encoding (kategorik integer kodlamasi)
train_tn = train.copy(); test_tn = test.copy()
cat_dims = []
for col in cat_cols:
    combined_v = pd.concat([train_tn[col], test_tn[col]], axis=0).astype(str).fillna('missing').replace('nan', 'missing')
    le = LabelEncoder()
    le.fit(combined_v)
    train_tn[col] = le.transform(train_tn[col].astype(str).fillna('missing').replace('nan', 'missing'))
    test_tn[col]  = le.transform(test_tn[col].astype(str).fillna('missing').replace('nan', 'missing'))
    cat_dims.append(len(le.classes_))

features_tn = [c for c in train_tn.columns if c not in ['id', TARGET]]
num_cols_tn = [c for c in features_tn if c not in cat_cols]
for c in num_cols_tn:
    med = train_tn[c].median()
    train_tn[c] = train_tn[c].fillna(med).replace([np.inf, -np.inf], med)
    test_tn[c]  = test_tn[c].fillna(med).replace([np.inf, -np.inf], med)

cat_idx_tn = [features_tn.index(c) for c in cat_cols]
X_tn = train_tn[features_tn].values.astype(np.float32)
X_test_tn = test_tn[features_tn].values.astype(np.float32)
y_tn = y.reshape(-1, 1).astype(np.float32)

oof_tn  = np.zeros(len(train))
pred_tn = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tn)):
    model_tn = TabNetRegressor(
        cat_idxs=cat_idx_tn,
        cat_dims=cat_dims,
        cat_emb_dim=5,
        n_d=16, n_a=16,
        n_steps=4,
        gamma=1.3,
        n_independent=2, n_shared=2,
        seed=SEED,
        verbose=0,
    )
    model_tn.fit(
        X_tn[tr_idx], y_tn[tr_idx],
        eval_set=[(X_tn[val_idx], y_tn[val_idx])],
        eval_name=['val'], eval_metric=['rmse'],
        max_epochs=30, patience=5,
        batch_size=2048, virtual_batch_size=256,
        num_workers=0, drop_last=False,
    )
    oof_tn[val_idx] = model_tn.predict(X_tn[val_idx]).flatten()
    pred_tn += model_tn.predict(X_test_tn).flatten() / N_SPLITS
    print(f"  Fold {fold+1}: RMSE = {np.sqrt(mean_squared_error(y[val_idx], oof_tn[val_idx])):.5f}")

print(f"  TabNet OOF RMSE: {np.sqrt(mean_squared_error(y, oof_tn)):.5f}")
print(f"  Korelasyon (v5 ile): {np.corrcoef(v5_oof, oof_tn)[0,1]:.5f}")

# Final mikro blend: optimal w_tn'i bul
best_w_tn, best_blend_rmse = 0.0, np.sqrt(mean_squared_error(y, v5_oof))
for w in np.arange(0.0, 0.151, 0.005):
    blend = (1 - w) * v5_oof + w * oof_tn
    r = np.sqrt(mean_squared_error(y, blend))
    if r < best_blend_rmse:
        best_blend_rmse, best_w_tn = r, w

print(f"\n  Optimal TabNet agirligi: {best_w_tn:.3f}")
print(f"  v5 + TabNet OOF RMSE: {best_blend_rmse:.5f}")

final_oof  = (1 - best_w_tn) * v5_oof + best_w_tn * oof_tn
final_pred = (1 - best_w_tn) * v5_pred + best_w_tn * pred_tn
final_pred = np.clip(final_pred, 0, 10)


# ============================================================================
# SONUC
# ============================================================================
print("\n" + "=" * 80)
print("FINAL METRIKLER")
print("=" * 80)
print(f"OOF RMSE  : {np.sqrt(mean_squared_error(y, final_oof)):.5f}")
print(f"OOF MAE   : {mean_absolute_error(y, final_oof):.5f}")
print(f"OOF R^2   : {r2_score(y, final_oof):.5f}")
print(f"Toplam sure: {(time.time() - T_START) / 60:.1f} dakika")

# Submission yaz
sub = pd.DataFrame({'id': test_ids, TARGET: final_pred})
sub_path = os.path.join(OUTPUT_DIR, 'submission.csv')
sub.to_csv(sub_path, index=False)
print(f"\nSubmission shape: {sub.shape}")
print(sub.head())
print(f"\nKaydedildi: {sub_path}")
print("=" * 80)
