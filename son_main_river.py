
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re
import os
import pickle

warnings.filterwarnings('ignore')

# River (online learning) imports - required on your machine
try:
    from river import anomaly, preprocessing
except Exception as e:
    # If river is not installed, we'll inform user in the app UI later.
    anomaly = None
    preprocessing = None

# ======================================================================
# CONFIG
# ======================================================================
st.set_page_config(
    page_title="Su Tüketim Davranış Analiz Dashboard (River Entegre)",
    page_icon="💧",
    layout="wide"
)

MODEL_PATH = "river_model.pkl"

# ======================================================================
# RIVER MODEL HELPERS (ONLINE LEARNING)
# ======================================================================
def river_available():
    return anomaly is not None and preprocessing is not None

def load_or_create_river_model():
    """
    Load existing River model from disk or create a new pipeline.
    The model is a pipeline: StandardScaler -> HalfSpaceTrees (anomaly detector)
    """
    if not river_available():
        return None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            st.info("🧠 River modeli yüklendi.")
            return model
        except Exception as e:
            st.warning(f"Model yüklenirken hata: {e}. Yeni model oluşturuluyor.")
    # Create new model
    model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(seed=42, n_estimators=40)
    st.info("🆕 Yeni River modeli oluşturuldu.")
    return model

def save_river_model(model):
    if model is None:
        return
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    st.success("💾 River modeli kaydedildi.")

def update_model_with_new_data(df, model, feature_map=None):
    """
    Update River model with rows in df. Returns scores list (higher = more anomalous for HalfSpaceTrees.score_one).
    feature_map: dict mapping expected feature names to df column names. If None, default mapping is used.
    """
    if model is None:
        raise RuntimeError("River modeli yok. river kütüphanesinin kurulu olduğundan emin olun.")

    if feature_map is None:
        feature_map = {
            "tuketim": "AKTIF_m3",
            "gunluk_ort": "GUNLUK_ORT_TUKETIM_m3",
            "tutar": "TOPLAM_TUTAR"
        }

    scores = []
    # iterate rows without building a big structure in memory
    for _, row in df.iterrows():
        try:
            x = {
                "tuketim": float(row.get(feature_map["tuketim"], 0.0)),
                "gunluk_ort": float(row.get(feature_map["gunluk_ort"], 0.0)),
                "tutar": float(row.get(feature_map["tutar"], 0.0))
            }
        except Exception:
            x = {k: 0.0 for k in feature_map.keys()}
        # compute score (higher -> more anomalous for HalfSpaceTrees.score_one)
        try:
            score = model.score_one(x)
        except Exception:
            score = 0.0
        # learn from this row (online)
        try:
            model.learn_one(x)
        except Exception:
            pass
        scores.append(score)
    # persist model
    save_river_model(model)
    return scores, model

# ======================================================================
# VERI İŞLEME FONKSİYONLARI (İKİ DOSYA OKUYAN)
# ======================================================================
@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder"""
    try:
        # Ana veri dosyasını oku
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None, None

    # Tarih formatını düzelt
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    
    # Tesisat numarası olan kayıtları filtrele
    df = df[df['TESISAT_NO'].notnull()]
    
    # Zone veri dosyasını oku
    kullanici_zone_verileri = {}
    if zone_file is not None:
        try:
            zone_excel_df = pd.read_excel(zone_file)
            st.success(f"✅ Zone veri dosyası başarıyla yüklendi: {len(zone_excel_df)} kayıt")
            
            # Zone verilerini işle
            for idx, row in zone_excel_df.iterrows():
                # Karne no ve adını ayır
                if 'KARNE NO VE ADI' in row:
                    karne_adi = str(row['KARNE NO VE ADI']).strip()
                    
                    # Karne numarasını çıkar (ilk 4 rakam)
                    karne_no_match = re.search(r'(\d{4})', karne_adi)
                    if karne_no_match:
                        karne_no = karne_no_match.group(1)
                        
                        # Zone bilgilerini topla
                        zone_bilgisi = {
                            'ad': karne_adi,
                            'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                            'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                            'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                        }
                        
                        kullanici_zone_verileri[karne_no] = zone_bilgisi
        except Exception as e:
            st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

    # Davranış analizi fonksiyonları
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Davranış analizi fonksiyonu (şüpheli tesisat tespiti)
    def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')

        if len(tesisat_verisi) < 3:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta"

        tuketimler = tesisat_verisi['AKTIF_m3'].values
        tarihler_series = tesisat_verisi['OKUMA_TARIHI']

        # Sıfır tüketim analizi
        sifir_sayisi = sum(tuketimler == 0)

        # Varyasyon analizi
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0

        # Trend analizi (son 3 dönem)
        if len(tuketimler) >= 3:
            son_uc = tuketimler[-3:]
            trend = "artış" if son_uc[2] > son_uc[0] * 1.2 else "azalış" if son_uc[2] < son_uc[0] * 0.8 else "stabil"
        else:
            trend = "belirsiz"

        # Şüpheli durum tespiti ve risk seviyesi
        suphe_aciklamasi = ""
        suphe_donemleri = []
        risk_seviyesi = "Düşük"

        # 1. Düzensiz sıfır tüketim paterni
        if sifir_sayisi >= 3:
            sifir_indisler = np.where(tuketimler == 0)[0]
            if len(sifir_indisler) >= 3:
                ardisik_olmayan = sum(np.diff(sifir_indisler) > 1) >= 2
                if ardisik_olmayan:
                    suphe_aciklamasi += "Düzensiz sıfır tüketim paterni. "
                    risk_seviyesi = "Yüksek"
                    for idx in sifir_indisler:
                        tarih_obj = pd.Timestamp(tarihler_series.iloc[idx])
                        suphe_donemleri.append(tarih_obj.strftime('%m/%Y'))

        # 2. Ani tüketim değişiklikleri
        if varyasyon_katsayisi > 1.5 and mean_tuketim > 5:
            suphe_aciklamasi += "Tüketimde yüksek dalgalanma. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi

        # 3. Trend analizi
        if trend == "artış" and mean_tuketim > 20:
            suphe_aciklamasi += "Yükselen tüketim trendi. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi

        # 4. Son dönem sıfır tüketim
        if tuketimler[-1] == 0 and len(tuketimler) > 1:
            suphe_aciklamasi += "Son dönem sıfır tüketim. "
            risk_seviyesi = "Yüksek" if sifir_sayisi >= 2 else "Orta"

        # Şüpheli dönemler varsa risk en az Orta olmalı
        if suphe_donemleri and risk_seviyesi == "Düşük":
            risk_seviyesi = "Orta"

        # Yorum kütüphanesi
        yorumlar_normal = ["Normal tüketim paterni", "Stabil tüketim alışkanlığı"]
        yorumlar_supheli = [
            "Tüketim alışkanlıklarında değişiklik gözlemleniyor",
            "Düzensiz tüketim paterni dikkat çekici",
            "Tüketim davranışında tutarsızlık mevcut",
            "Değişken tüketim alışkanlıkları",
            "Tüketim paterninde olağandışı dalgalanma"
        ]

        if not suphe_aciklamasi:
            davranis_yorumu = np.random.choice(yorumlar_normal)
        else:
            davranis_yorumu = np.random.choice(yorumlar_supheli)

        return davranis_yorumu, ", ".join(suphe_donemleri) if suphe_donemleri else "Yok", risk_seviyesi

    # Tüm tesisatlar için davranış analizi yap
    davranis_sonuclari = []
    for i, (idx, row) in enumerate(son_okumalar.iterrows()):
        yorum, supheli_donemler, risk = tesisat_davranis_analizi(row['TESISAT_NO'], row, df)
        davranis_sonuclari.append({
            'TESISAT_NO': row['TESISAT_NO'],
            'DAVRANIS_YORUMU': yorum,
            'SUPHELI_DONEMLER': supheli_donemler,
            'RISK_SEVIYESI': risk
        })

    davranis_df = pd.DataFrame(davranis_sonuclari)
    son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    # Zone analizi
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        ekim_2024_df = df[(df['OKUMA_TARIHI'].dt.month == 10) & (df['OKUMA_TARIHI'].dt.year == 2024)]
        if len(ekim_2024_df) == 0:
            ekim_2024_df = df.copy()
        
        zone_analizi = ekim_2024_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        # Zone risk analizi
        ekim_2024_risk = ekim_2024_df.merge(son_okumalar[['TESISAT_NO', 'RISK_SEVIYESI']], on='TESISAT_NO', how='left')
        zone_risk_analizi = ekim_2024_risk.groupby('KARNE_NO')['RISK_SEVIYESI'].apply(
            lambda x: (x == 'Yüksek').sum() if 'Yüksek' in x.values else 0
        ).reset_index(name='YUKSEK_RISKLI_TESISAT')

        zone_analizi = zone_analizi.merge(zone_risk_analizi, on='KARNE_NO', how='left')
        zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100

        # Kullanıcı zone verilerini birleştir
        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi, kullanici_zone_verileri

# ======================================================================
# STREAMLIT ARAYÜZ
# ======================================================================
st.title("💧 Su Tüketim Davranış Analiz Dashboard (River Entegre)")

# Sidebar - dosya yükleme
st.sidebar.header("📁 Veri ve Model")
uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin (yavuz.xlsx)",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin (yavuzeli merkez ekim.xlsx)",
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Yeni veri için incremental uploader (CSV veya XLSX)
incremental_file = st.sidebar.file_uploader(
    "Yeni veri yükle (incremental CSV/XLSX) - model bu verilerle satır satır öğrenecek",
    type=["csv", "xlsx"]
)

# Model yönetimi
model = load_or_create_river_model() if river_available() else None
if not river_available():
    st.sidebar.warning("River kütüphanesi yüklenmemiş. Online öğrenme çalışmaz. `pip install river` yapın.")

if st.sidebar.button("🔁 Modeli Sıfırla (sil)"):
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        model = load_or_create_river_model()
        st.sidebar.success("Model sıfırlandı.")
    except Exception as e:
        st.sidebar.error(f"Model sıfırlanırken hata: {e}")

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
    demo_data = []
    for i in range(1000):
        tesisat_no = f"TS{1000 + i}"
        aktif_m3 = np.random.gamma(2, 10)
        toplam_tutar = aktif_m3 * 15 + np.random.normal(0, 10)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0.1),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
        })
    
    df = pd.DataFrame(demo_data)
    son_okumalar = df.copy()
    son_okumalar['OKUMA_PERIYODU_GUN'] = 300
    son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
    son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
    
    # Heuristic risk (demo)
    risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.7, 0.2, 0.1])
    son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
    son_okumalar['DAVRANIS_YORUMU'] = "Demo verisi - analiz edildi"
    son_okumalar['SUPHELI_DONEMLER'] = "Yok"
    
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    st.success("✅ Demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Eğer incremental veri yüklendiyse, modeli satır satır güncelle ve skor üret
if incremental_file is not None and model is not None:
    st.sidebar.info("Yeni veri işlendi: model bu verilerle öğreniyor...")
    try:
        if incremental_file.name.lower().endswith(".csv"):
            inc_df = pd.read_csv(incremental_file)
        else:
            inc_df = pd.read_excel(incremental_file)
        # Eğer OKUMA tarihleri veya ILK_OKUMA yoksa, basit doldurma
        if 'OKUMA_TARIHI' in inc_df.columns and not pd.api.types.is_datetime64_any_dtype(inc_df['OKUMA_TARIHI']):
            inc_df['OKUMA_TARIHI'] = pd.to_datetime(inc_df['OKUMA_TARIHI'], errors='coerce')
        if 'ILK_OKUMA_TARIHI' in inc_df.columns and not pd.api.types.is_datetime64_any_dtype(inc_df['ILK_OKUMA_TARIHI']):
            inc_df['ILK_OKUMA_TARIHI'] = pd.to_datetime(inc_df['ILK_OKUMA_TARIHI'], errors='coerce')
        # Eğer günlük ortalama yoksa hesapla (basit)
        if 'GUNLUK_ORT_TUKETIM_m3' not in inc_df.columns and 'OKUMA_PERIYODU_GUN' in inc_df.columns:
            inc_df['GUNLUK_ORT_TUKETIM_m3'] = inc_df['AKTIF_m3'] / inc_df['OKUMA_PERIYODU_GUN'].clip(lower=1)
        elif 'GUNLUK_ORT_TUKETIM_m3' not in inc_df.columns:
            inc_df['GUNLUK_ORT_TUKETIM_m3'] = inc_df['AKTIF_m3']  # fallback
        
        scores, model = update_model_with_new_data(inc_df, model)
        inc_df['RIVER_SCORE'] = scores
        st.sidebar.success(f"🧠 Model {len(inc_df)} satırı öğrendi. Son skorlardan örnek: {inc_df['RIVER_SCORE'].head().tolist()}")
        
        # Merge incremental scores into son_okumalar where TESISAT_NO matches last record
        # For simplicity, take mean score per TESISAT_NO
        try:
            mean_scores = inc_df.groupby('TESISAT_NO')['RIVER_SCORE'].mean().reset_index().rename(columns={'RIVER_SCORE': 'RIVER_SCORE_MEAN'})
            son_okumalar = son_okumalar.merge(mean_scores, on='TESISAT_NO', how='left')
        except Exception:
            pass
    except Exception as e:
        st.sidebar.error(f"Yeni veri işlerken hata: {e}")

# Genel Metrikler
if son_okumalar is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Toplam Tesisat",
            value=f"{len(son_okumalar):,}"
        )
    
    with col2:
        st.metric(
            label="💧 Toplam Tüketim",
            value=f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³"
        )
    
    with col3:
        st.metric(
            label="💰 Toplam Gelir",
            value=f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL"
        )
    
    with col4:
        # Risk dağılımı
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric(
            label="🚨 Yüksek Riskli Tesisat (Heuristik)",
            value=f"{yuksek_riskli}"
        )

# Tab Menü (görselleştirme temel yapı aynı)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Genel Görünüm", 
    "🗺️ Zone Analizi", 
    "🔍 Detaylı Analiz", 
    "📊 İleri Analiz",
    "🔥 Ateş Böceği Görünümü"
])

with tab1:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                              title='Günlük Tüketim Dağılımı',
                              labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'},
                              color_discrete_sequence=['#3498DB'])
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            color_col = 'RISK_SEVIYESI' if 'RISK_SEVIYESI' in son_okumalar.columns else None
            fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            color=color_col,
                            title='Tüketim-Tutar İlişkisi (Risk Seviyeli)',
                            labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                            color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig2, use_container_width=True)
        
        if 'RIVER_SCORE_MEAN' in son_okumalar.columns:
            st.markdown("**🧠 River (online) tarafından hesaplanan ortalama anomali skorları (tesisat bazında)**")
            st.dataframe(son_okumalar[['TESISAT_NO', 'RIVER_SCORE_MEAN']].sort_values('RIVER_SCORE_MEAN', ascending=False).head(20), use_container_width=True)

with tab2:
    if zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                        title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            fig5 = px.bar(zone_analizi, x='KARNE_NO', y='TESISAT_SAYISI',
                        title='Zone Bazlı Tesisat Sayısı',
                        labels={'KARNE_NO': 'Zone', 'TESISAT_SAYISI': 'Tesisat Sayısı'},
                        color_discrete_sequence=['#E74C3C'])
            st.plotly_chart(fig5, use_container_width=True)
        
        st.subheader("Zone Karşılaştırma Tablosu")
        zone_karsilastirma = zone_analizi[['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR', 'YUKSEK_RISK_ORANI']].copy()
        if 'ad' in zone_analizi.columns:
            zone_karsilastirma['Zone Adı'] = zone_analizi['ad']
        if 'verilen_su' in zone_analizi.columns:
            zone_karsilastirma['Verilen Su (m³)'] = zone_analizi['verilen_su']
            zone_karsilastirma['Tahakkuk (m³)'] = zone_analizi['tahakkuk_m3']
            zone_karsilastirma['Kayıp Oranı (%)'] = zone_analizi['kayip_oran']
        
        st.dataframe(zone_karsilastirma, use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    if son_okumalar is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Filtreleme Seçenekleri")
            
            tuketim_range = st.slider(
                "Tüketim Aralığı (m³)",
                min_value=0,
                max_value=int(son_okumalar['AKTIF_m3'].max()) if len(son_okumalar) > 0 else 100,
                value=[0, 100],
                help="Tüketim değerine göre filtreleme yapın"
            )
            
            risk_seviyeleri = st.multiselect(
                "Risk Seviyeleri",
                options=['Düşük', 'Orta', 'Yüksek'],
                default=['Yüksek', 'Orta']
            )
            
            siralama = st.selectbox(
                "Sıralama Türü",
                options=['En Yüksek Tüketim', 'En Düşük Tüketim', 'En Yüksek Risk'],
                index=2
            )
        
        with col2:
            min_tuketim, max_tuketim = tuketim_range
            filtreli_veri = son_okumalar[
                (son_okumalar['AKTIF_m3'] >= min_tuketim) & 
                (son_okumalar['AKTIF_m3'] <= max_tuketim) &
                (son_okumalar['RISK_SEVIYESI'].isin(risk_seviyeleri))
            ]
            
            if siralama == 'En Yüksek Tüketim':
                gosterilecek_veri = filtreli_veri.nlargest(20, 'AKTIF_m3')
            elif siralama == 'En Düşük Tüketim':
                gosterilecek_veri = filtreli_veri.nsmallest(20, 'AKTIF_m3')
            else:
                risk_sirasi = {'Yüksek': 3, 'Orta': 2, 'Düşük': 1}
                filtreli_veri['RISK_SIRASI'] = filtreli_veri['RISK_SEVIYESI'].map(risk_sirasi)
                gosterilecek_veri = filtreli_veri.nlargest(20, ['RISK_SIRASI', 'AKTIF_m3'])
            
            st.dataframe(
                gosterilecek_veri[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'GUNLUK_ORT_TUKETIM_m3', 'RISK_SEVIYESI', 'DAVRANIS_YORUMU']].round(3),
                use_container_width=True
            )

with tab4:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
            fig6 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                         title='Risk Seviyeleri Dağılımı',
                         color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            numeric_cols = son_okumalar.select_dtypes(include=[np.number]).columns
            corr_matrix = son_okumalar[numeric_cols].corr()
            
            fig7 = px.imshow(corr_matrix, 
                           title='Korelasyon Matrisi',
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
            st.plotly_chart(fig7, use_container_width=True)
        
        fig8 = px.box(son_okumalar, y='AKTIF_m3', 
                     title='Tüketim Dağılımı - Aykırı Değer Analizi',
                     color_discrete_sequence=['#F39C12'])
        st.plotly_chart(fig8, use_container_width=True)
        
        if 'RIVER_SCORE_MEAN' in son_okumalar.columns:
            fig_anom = px.histogram(
                son_okumalar,
                x='RIVER_SCORE_MEAN',
                nbins=50,
                title='River Anomali Skoru Dağılımı (Tesisat Bazlı Ortalama)'
            )
            st.plotly_chart(fig_anom, use_container_width=True)

with tab5:
    st.header("🔥 Ateş Böceği Görünümü - Şüpheli Tesisatlar (Heuristik ve River)")
    
    if son_okumalar is not None:
        # Combine heuristic high risk and river high score
        # Define high river score threshold heuristically (you can tune later)
        if 'RIVER_SCORE_MEAN' in son_okumalar.columns:
            son_okumalar['RIVER_ANOMALY_FLAG'] = son_okumalar['RIVER_SCORE_MEAN'] > 0.5  # tuneable
        else:
            son_okumalar['RIVER_ANOMALY_FLAG'] = False
        
        # Yüksek riskli (heuristic)
        yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
        # Ayrıca river ile belirlenen şüpheliler
        river_supheli = son_okumalar[son_okumalar['RIVER_ANOMALY_FLAG'] == True]
        
        combined_supheli = pd.concat([yuksek_riskli, river_supheli]).drop_duplicates('TESISAT_NO')
        
        if len(combined_supheli) > 0:
            st.success(f"🚨 {len(combined_supheli)} adet yüksek riskli/şüpheli tesisat tespit edildi!")
            fig9 = px.scatter(combined_supheli, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            size='GUNLUK_ORT_TUKETIM_m3',
                            color='RIVER_ANOMALY_FLAG' if 'RIVER_ANOMALY_FLAG' in combined_supheli.columns else 'RISK_SEVIYESI',
                            hover_name='TESISAT_NO',
                            title='🔥 Ateş Böceği Görünümü - Model Destekli',
                            labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'})
            fig9.update_traces(marker=dict(symbol='star', line=dict(width=2)))
            st.plotly_chart(fig9, use_container_width=True)
            
            for idx, row in combined_supheli.iterrows():
                with st.expander(f"🚨 Tesisat No: {row['TESISAT_NO']} - {row.get('DAVRANIS_YORUMU','')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tüketim", f"{row['AKTIF_m3']:.1f} m³")
                    with col2:
                        st.metric("Tutar", f"{row['TOPLAM_TUTAR']:.1f} TL")
                    with col3:
                        st.metric("Günlük Ort.", f"{row['GUNLUK_ORT_TUKETIM_m3']:.3f} m³")
                    
                    st.write(f"**Şüpheli Dönemler:** {row.get('SUPHELI_DONEMLER','Yok')}")
                    st.write(f"**Davranış Yorumu:** {row.get('DAVRANIS_YORUMU','')}")
        else:
            st.info("🎉 Hiç yüksek riskli tesisat bulunamadı!")

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim Analiz Sistemi | Streamlit Dashboard | 🧠 River - Online Learning Entegre")
