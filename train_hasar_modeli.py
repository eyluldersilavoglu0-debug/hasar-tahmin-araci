# -*- coding: utf-8 -*-
"""
Hafta 6 Ödevi — Hasar Tahmin Modeli Eğitimi (şablon)

Bu dosya Colab'da çalıştırılır:  !python train_hasar_modeli.py
Gerekli dosya: bina_hasar_verileri_500.csv (aynı klasörde olmalı)
Çıktı        : models/hasar_modeli.joblib  ve  models/meta.json

Yalnızca "1. ÖĞRENCİNİN DEĞİŞTİRECEĞİ BÖLÜM" kısmını düzenleyin.
Diğer bölümler uygulamanın (app.py) modeli doğru okuyabilmesi için gereklidir;
değiştirirseniz uygulama çalışmayabilir.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# ======================================================================
# 1. ÖĞRENCİNİN DEĞİŞTİRECEĞİ BÖLÜM
# ======================================================================
# Her deneme için bu değerleri değiştirip dosyayı yeniden çalıştırın.
# Raporunuza son seçtiğiniz değerleri ve nedenini yazın.

N_ESTIMATORS = 100      # Ormandaki karar ağacı sayısı (örn. 50, 100, 300)
MAX_DEPTH = None        # Ağaç derinliği sınırı: None (sınırsız) ya da 4, 6, 8, 12 ...
MIN_SAMPLES_LEAF = 1    # Bir yaprakta en az kaç bina kalsın (örn. 1, 2, 5, 10)
CLASS_WEIGHT = None     # None ya da 'balanced' (az görülen sınıflara ağırlık verir)

# ======================================================================
# 2. SABİT BÖLÜM (değiştirmeyin)
# ======================================================================
DATA_PATH = "bina_hasar_verileri_500.csv"
MODEL_DIR = "models"
SIRA = ["Hasarsız", "Hafif", "Orta", "Ağır", "Yıkık"]   # sınıf sırası (0-4)
GOZLEM_DISI = ["bina_no", "hasar_seviyesi", "hasar_etiketi"]
MAX_DOSYA_MB = 90   # GitHub tek dosya sınırı 100 MB

df = pd.read_csv(DATA_PATH)
print(f"Veri: {df.shape[0]} bina, {df.shape[1]} sütun\n")

# --- Çatlak yoğunluğunun sınıflara göre dağılımı --------------------
# Uygulamadaki "çatlak kuralı" bu ölçeğe uygun bir sayı (0-1 arası) üretmelidir.
print("Çatlak yoğunluğu (catl_yogunluk) — hasar sınıfına göre:")
print(df.groupby("hasar_etiketi")["catl_yogunluk"]
        .describe()[["mean", "min", "25%", "50%", "75%", "max"]]
        .reindex(SIRA).round(2))
print()

# --- Özellik hazırlama (Hafta 6 lab ile aynı) -----------------------
df_m = pd.get_dummies(df, columns=["yapi_turu", "deprem_bolgesi"],
                      drop_first=True, dtype=float)
df_m["yonetmelik_oncesi"] = (df_m["insaat_yili"] < 1999).astype(int)
X = df_m.drop(columns=GOZLEM_DISI)
y = df_m["hasar_seviyesi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model ----------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF, class_weight=CLASS_WEIGHT,
    random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc_train = accuracy_score(y_train, rf.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_skor = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
recall = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

print("=== SEÇTİĞİNİZ PARAMETRELER ===")
print(f"n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
      f"min_samples_leaf={MIN_SAMPLES_LEAF}, class_weight={CLASS_WEIGHT}\n")
print("=== SONUÇLAR (raporda kullanacaksınız) ===")
print(f"Eğitim doğruluğu      : {acc_train:.3f}")
print(f"Test doğruluğu        : {acc_test:.3f}")
print(f"5 katlı CV doğruluğu  : {cv_skor.mean():.3f} (std {cv_skor.std():.3f})")
for s, r in zip(SIRA, recall):
    print(f"  {s:9s} recall: {r:.2f}")
print()
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4],
                            target_names=SIRA, zero_division=0))
print("Karmaşıklık matrisi (satır: gerçek, sütun: tahmin):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4]))
print()

# --- Kaydetme -------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model_yolu = os.path.join(MODEL_DIR, "hasar_modeli.joblib")
joblib.dump(rf, model_yolu, compress=3)
mb = os.path.getsize(model_yolu) / 1e6
print(f"Model dosyası: {model_yolu}  ({mb:.1f} MB)")
if mb > MAX_DOSYA_MB:
    print("UYARI: Dosya GitHub sınırına yaklaştı. N_ESTIMATORS değerini "
          "azaltın ya da MAX_DEPTH / MIN_SAMPLES_LEAF ile ağaçları küçültün.")
    sys.exit(1)

# Uygulamanın formu ve uyarıları için gerekli bilgiler
sayisal = ["kat_sayisi", "insaat_yili", "pga_g", "catl_yogunluk",
           "deformasyon_yuzde", "doseme_hasari", "kolon_hasari"]
meta = {
    "ozellikler": list(X.columns),                      # modelin beklediği sütun sırası
    "yapi_turleri": sorted(df["yapi_turu"].unique().tolist()),
    "deprem_bolgeleri": sorted(df["deprem_bolgesi"].unique().tolist()),
    "aralik": {c: [float(df[c].min()), float(df[c].max())] for c in sayisal},
    "sira": SIRA,
    "parametreler": {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
                     "min_samples_leaf": MIN_SAMPLES_LEAF, "class_weight": CLASS_WEIGHT},
    "metrikler": {"egitim_dogrulugu": round(float(acc_train), 4),
                  "test_dogrulugu": round(float(acc_test), 4),
                  "cv_dogrulugu": round(float(cv_skor.mean()), 4),
                  "recall": {s: round(float(r), 3) for s, r in zip(SIRA, recall)}},
    "sklearn_surumu": sklearn.__version__,
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("meta.json yazıldı.")
print("\nSıradaki adım: models/ klasöründeki 2 dosyayı bilgisayarınıza indirin.")
