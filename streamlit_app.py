import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re

warnings.filterwarnings('ignore')

# Adaptive model import
try:
    from adaptive_model_api import adaptive_model
except ImportError:
    st.error("Adaptive Model API yüklenemedi. Lütfen adaptive_model_api.py dosyasını kontrol edin.")
    st.stop()

# ======================================================================
# 🧠 HEMEN ÖĞRENEN ADAPTIVE STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Hemen Öğrenen Su Tüketim AI",
    page_icon="🚀",
    layout="wide"
)

# ======================================================================
# 📊 AKILLI VERİ İŞLEME FONKSİYONLARI - 1M+ SATIR DESTEKLİ
# ======================================================================

@st.cache_data(ttl=3600, max_entries=2, show_spinner="Büyük veri seti yükleniyor...")
def load_million_plus_data(uploaded_file, sampling_ratio=None):
    """1M+ satır için optimize veri yükleme"""
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > 50:  # 50MB'tan büyük dosya
        st.warning(f"📁 Büyük dosya tespit edildi: {file_size_mb:.1f} MB")
        
        if sampling_ratio is None:
            # Otomatik örnekleme oranı belirle
            if file_size_mb > 200:
                sampling_ratio = 0.1  # 200MB+ → %10
            elif file_size_mb > 100:
                sampling_ratio = 0.2  # 100-200MB → %20
            else:
                sampling_ratio = 0.3  # 50-100MB → %30
        
        st.info(f"🎯 %{sampling_ratio*100} örnekleme ile {sampling_ratio:.0%} veri analiz edilecek")
        
        # Akıllı örnekleme
        try:
            # Chunk'lar halinde oku ve örnekle
            chunk_size = 50000
            chunks = pd.read_excel(uploaded_file, chunksize=chunk_size, dtype={'TESISAT_NO': str})
            
            sampled_chunks = []
            for chunk in chunks:
                if np.random.random() < sampling_ratio:
                    sampled_chunks.append(chunk)
                
                # Maksimum 500K satır (performans için)
                total_sampled = sum(len(c) for c in sampled_chunks)
                if total_sampled > 500000:
                    st.info("⏹️ 500K satır limite ulaşıldı")
                    break
            
            df = pd.concat(sampled_chunks, ignore_index=True)
            st.success(f"✅ {len(df):,} satır örneklenerek yüklendi")
            
        except Exception as e:
            st.error(f"❌ Örnekleme hatası: {e}")
            # Fallback: direkt okuma
            df = pd.read_excel(uploaded_file, dtype={'TESISAT_NO': str})
            df = df.sample(frac=min(sampling_ratio, 0.3))  # Güvenli örnekleme
    
    else:
        # Küçük dosya - direkt yükle
        df = pd.read_excel(uploaded_file, dtype={'TESISAT_NO': str})
    
    return df

def calculate_realistic_consumption(df):
    """GERÇEKÇİ günlük tüketim hesaplama - KRİTİK DÜZELTME"""
    # Her okuma bir AY'lık tüketim! Bu yüzden:
    # Günlük ortalama = Aylık tüketim / 30 gün
    
    # Önce temel periyot (30 gün - standart ay)
    df['OKUMA_PERIYODU_GUN'] = 30
    
    # Tarih farkına göre daha doğru periyot (opsiyonel)
    mask = df['OKUMA_TARIHI'].notna() & df['ILK_OKUMA_TARIHI'].notna()
    df.loc[mask, 'OKUMA_PERIYODU_GUN'] = (df.loc[mask, 'OKUMA_TARIHI'] - df.loc[mask, 'ILK_OKUMA_TARIHI']).dt.days
    
    # Gerçekçi periyot sınırları (25-35 gün arası)
    df['OKUMA_PERIYODU_GUN'] = df['OKUMA_PERIYODU_GUN'].clip(lower=25, upper=35)
    
    # GERÇEK günlük ortalama = Aylık tüketim / gün sayısı
    df['GUNLUK_ORT_TUKETIM_m3'] = df['AKTIF_m3'] / df['OKUMA_PERIYODU_GUN']
    
    # Gerçekçi sınırlar (ev/işyeri tüketimi için)
    # Günlük 50m³'den fazla şüpheli, maksimum 100m³
    df['GUNLUK_ORT_TUKETIM_m3'] = df['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
    
    return df

@st.cache_data(ttl=3600)
def load_and_analyze_data_adaptive(uploaded_file, zone_file):
    """Adaptive analiz ile veri işleme - 1M+ SATIR DESTEKLİ"""
    try:
        # Büyük veri yükleme
        df = load_million_plus_data(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df):,} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None, None

    # Tesisat numarasını temizle
    def clean_tesisat_no(tesisat_no):
        if pd.isna(tesisat_no):
            return None
        tesisat_str = str(tesisat_no).strip()
        tesisat_str = re.sub(r'[,"\']', '', tesisat_str)
        return tesisat_str.strip()

    df['TESISAT_NO'] = df['TESISAT_NO'].apply(clean_tesisat_no)
    df = df[df['TESISAT_NO'].notnull()]
    df = df[df['TESISAT_NO'] != '']
    df = df[df['TESISAT_NO'] != 'nan']
    
    # Tarih ve sayısal sütunları temizle
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
    
    numeric_columns = ['AKTIF_m3', 'TOPLAM_TUTAR']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # KRİTİK DÜZELTME: Gerçekçi günlük tüketim hesaplama
    df = calculate_realistic_consumption(df)
    
    # Zone veri dosyasını oku
    kullanici_zone_verileri = {}
    if zone_file is not None:
        try:
            zone_excel_df = pd.read_excel(zone_file)
            st.success(f"✅ Zone veri dosyası başarıyla yüklendi: {len(zone_excel_df)} kayıt")
            
            for idx, row in zone_excel_df.iterrows():
                if 'KARNE NO VE ADI' in row:
                    karne_adi = str(row['KARNE NO VE ADI']).strip()
                    karne_no_match = re.search(r'(\d{4})', karne_adi)
                    if karne_no_match:
                        karne_no = karne_no_match.group(1)
                        zone_bilgisi = {
                            'ad': karne_adi,
                            'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                            'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                            'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                        }
                        kullanici_zone_verileri[karne_no] = zone_bilgisi
        except Exception as e:
            st.warning(f"⚠️ Zone veri dosyası yüklenirken hata: {e}")

    # Son okumaları al
    son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()

    # ADAPTIVE analiz yap - OPTİMİZE
    if len(son_okumalar) > 0:
        st.info("🧠 Adaptive AI analizi yapılıyor ve ÖĞRENİYOR...")
        progress_bar = st.progress(0)
        davranis_sonuclari = []
        
        total_tesisat = len(son_okumalar)
        
        # Büyük veri setleri için batch işleme
        batch_size = 100
        for batch_start in range(0, total_tesisat, batch_size):
            batch_end = min(batch_start + batch_size, total_tesisat)
            batch_data = son_okumalar.iloc[batch_start:batch_end]
            
            for i, (idx, row) in enumerate(batch_data.iterrows()):
                tesisat_verisi = df[df['TESISAT_NO'] == row['TESISAT_NO']].sort_values('OKUMA_TARIHI')
                
                # Adaptive analiz - ÖĞRENME ENTEGRE
                analiz_sonucu = adaptive_model.gelismis_davranis_analizi(tesisat_verisi)
                
                davranis_sonuclari.append({
                    'TESISAT_NO': row['TESISAT_NO'],
                    'DAVRANIS_YORUMU': analiz_sonucu['yorum'],
                    'SUPHELI_DONEMLER': analiz_sonucu['supheli_donemler'],
                    'RISK_SEVIYESI': analiz_sonucu['risk_seviyesi'],
                    'RISK_PUANI': analiz_sonucu['risk_puan'],
                    'PATTERN_DATA': analiz_sonucu.get('pattern_data', {})
                })
            
            # İlerleme güncelleme
            progress = min((batch_end) / total_tesisat, 1.0)
            progress_bar.progress(progress)

        progress_bar.progress(1.0)
        davranis_df = pd.DataFrame(davranis_sonuclari)
        son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    # Zone analizi
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        son_tarih = df['OKUMA_TARIHI'].max() if 'OKUMA_TARIHI' in df.columns else pd.Timestamp.now()
        uc_ay_once = son_tarih - timedelta(days=90)
        son_uc_ay_df = df[df['OKUMA_TARIHI'] >= uc_ay_once] if 'OKUMA_TARIHI' in df.columns else df
        
        if len(son_uc_ay_df) == 0:
            son_uc_ay_df = df.copy()
        
        zone_analizi = son_uc_ay_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        if 'RISK_SEVIYESI' in son_okumalar.columns:
            son_uc_ay_risk = son_uc_ay_df.merge(son_okumalar[['TESISAT_NO', 'RISK_SEVIYESI']], on='TESISAT_NO', how='left')
            
            zone_risk_analizi = son_uc_ay_risk.groupby('KARNE_NO').agg({
                'RISK_SEVIYESI': lambda x: (x == 'Yüksek').sum() if x.notna().any() else 0,
                'TESISAT_NO': 'count'
            }).reset_index()
            zone_risk_analizi.columns = ['KARNE_NO', 'YUKSEK_RISKLI_TESISAT', 'TOPLAM_TESISAT']
            
            zone_analizi = zone_analizi.merge(zone_risk_analizi[['KARNE_NO', 'YUKSEK_RISKLI_TESISAT']], on='KARNE_NO', how='left')
            zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100
            zone_analizi['YUKSEK_RISK_ORANI'] = zone_analizi['YUKSEK_RISK_ORANI'].fillna(0)

        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi, kullanici_zone_verileri

# ======================================================================
# 🎨 HEMEN ÖĞRENEN STREAMLIT ARAYÜZ
# ======================================================================

st.title("🚀 Hemen Öğrenen Su Tüketim AI Dashboard")
st.markdown("**Sentetik Veri ile Eğitilmiş & Aktif Öğrenen Yapay Zeka**")

# Sidebar - Gelişmiş Öğrenme Kontrolleri
st.sidebar.header("🧠 AI Öğrenme Kontrolleri")

# Öğrenme istatistikleri - GÜNCELLENMİŞ
learning_stats = adaptive_model.get_learning_stats()
st.sidebar.metric("🤖 Toplam Gözlem", f"{learning_stats['toplam_gozlem']:,}")
st.sidebar.metric("🎯 Gerçek Gözlem", f"{learning_stats['gercek_gozlem']:,}")
st.sidebar.metric("📊 Başarı Oranı", f"{learning_stats['basari_orani']:.1%}")

# Hızlı geri bildirim sistemi
st.sidebar.subheader("⚡ Hızlı Geri Bildirim")

# Otomatik geri bildirim butonları
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("👍 Doğru Tahmin"):
        adaptive_model.learn_from_feedback(
            "AUTO_POSITIVE", "Yüksek", "Yüksek", 
            {"type": "positive_feedback", "source": "auto"}
        )
        st.sidebar.success("✅ Olumlu feedback eklendi!")
        st.rerun()
        
with col2:
    if st.button("👎 Yanlış Tahmin"):
        adaptive_model.learn_from_feedback(
            "AUTO_NEGATIVE", "Düşük", "Yüksek", 
            {"type": "negative_feedback", "source": "auto"}
        )
        st.sidebar.warning("⚠️ Düzeltme feedback'i eklendi!")
        st.rerun()

with col3:
    if st.button("🔄 Modeli Yenile"):
        st.rerun()

# Manuel geri bildirim
st.sidebar.subheader("📝 Manuel Geri Bildirim")
feedback_tesisat = st.sidebar.text_input("Tesisat No", "TEST_001")
feedback_gercek = st.sidebar.selectbox("Gerçek Durum", ["Yüksek", "Orta", "Düşük"], index=0)
feedback_tahmin = st.sidebar.selectbox("AI Tahmini", ["Yüksek", "Orta", "Düşük"], index=0)

if st.sidebar.button("📤 Geri Bildirim Gönder"):
    if feedback_tesisat:
        adaptive_model.learn_from_feedback(
            feedback_tesisat, feedback_gercek, feedback_tahmin,
            {"type": "manual_feedback", "timestamp": datetime.now()}
        )
        st.sidebar.success("✅ Geri bildirim kaydedildi! AI öğreniyor...")
        st.rerun()

# Adaptive threshold'ları göster
st.sidebar.subheader("🔧 Adaptive Threshold'lar")
for key, value in learning_stats['adaptive_thresholds'].items():
    st.sidebar.write(f"**{key}**: `{value:.2f}`")

# Dosya yükleme bölümü
st.sidebar.header("📁 Dosya Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin (opsiyonel)",
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Gelişmiş Demo butonu
if st.sidebar.button("🎮 Gelişmiş Demo Modu"):
    st.info("🚀 Gelişmiş Demo modu aktif! AI hem analiz ediyor hem de ÖĞRENİYOR...")
    
    # Basit demo verisi oluştur
    np.random.seed(42)
    demo_data = []
    
    for i in range(200):
        tesisat_no = f"8000{300 + i}"
        pattern_type = np.random.choice(['normal', 'sifir_aralikli', 'yuksek'], p=[0.7, 0.2, 0.1])
        
        if pattern_type == 'normal':
            aktif_m3 = np.random.gamma(2, 8)
        elif pattern_type == 'sifir_aralikli':
            aktif_m3 = 0 if np.random.random() < 0.3 else np.random.gamma(2, 6)
        else:
            aktif_m3 = np.random.gamma(5, 15)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0),
            'TOPLAM_TUTAR': max(aktif_m3 * 15, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 4)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Çoklu okuma verisi
    coklu_okuma_data = []
    for tesisat in demo_data:
        for month in range(3):
            month_date = pd.Timestamp(f'2024-{8+month:02d}-15')
            consumption = tesisat['AKTIF_m3'] * (1 + np.random.normal(0, 0.3))
            
            coklu_okuma_data.append({
                'TESISAT_NO': tesisat['TESISAT_NO'],
                'AKTIF_m3': max(consumption, 0),
                'TOPLAM_TUTAR': max(consumption * 15, 0),
                'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
                'OKUMA_TARIHI': month_date,
                'KARNE_NO': tesisat['KARNE_NO']
            })
    
    df_detayli = pd.DataFrame(coklu_okuma_data)
    
    # Sayısal sütunları temizle
    df_detayli['AKTIF_m3'] = pd.to_numeric(df_detayli['AKTIF_m3'], errors='coerce')
    df_detayli['TOPLAM_TUTAR'] = pd.to_numeric(df_detayli['TOPLAM_TUTAR'], errors='coerce')
    
    # GERÇEKÇİ günlük tüketim hesaplama
    df_detayli = calculate_realistic_consumption(df_detayli)
    
    # Son okumaları al
    son_okumalar = df_detayli.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
    
    # Adaptive analiz - ÖĞRENME ENTEGRE
    davranis_sonuclari = []
    for tesisat_no in son_okumalar['TESISAT_NO'].unique():
        tesisat_verisi = df_detayli[df_detayli['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
        analiz_sonucu = adaptive_model.gelismis_davranis_analizi(tesisat_verisi)
        
        davranis_sonuclari.append({
            'TESISAT_NO': tesisat_no,
            'DAVRANIS_YORUMU': analiz_sonucu['yorum'],
            'SUPHEli_DONEMLER': analiz_sonucu['supheli_donemler'],
            'RISK_SEVIYESI': analiz_sonucu['risk_seviyesi'],
            'RISK_PUANI': analiz_sonucu['risk_puan'],
            'PATTERN_DATA': analiz_sonucu.get('pattern_data', {})
        })
    
    davranis_df = pd.DataFrame(davranis_sonuclari)
    son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')
    
    # Zone analizi
    zone_analizi = df_detayli.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    
    kullanici_zone_verileri = {
        'ZONE1': {'ad': 'BÖLGE-1', 'verilen_su': 10000, 'tahakkuk_m3': 7000, 'kayip_oran': 30.0},
        'ZONE2': {'ad': 'BÖLGE-2', 'verilen_su': 8000, 'tahakkuk_m3': 6000, 'kayip_oran': 25.0},
    }
    
    st.success("✅ Gelişmiş demo verisi oluşturuldu! AI analiz ediyor ve ÖĞRENİYOR.")

elif uploaded_file is not None:
    try:
        df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data_adaptive(uploaded_file, zone_file)
        if df is None:
            st.stop()
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        st.stop()
else:
    st.warning("⚠️ Lütfen Excel dosyasını yükleyin veya Gelişmiş Demo modunu kullanın")
    st.stop()

# Genel Metrikler - GÜNCELLENMİŞ
if son_okumalar is not None and len(son_okumalar) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Toplam Tesisat", f"{len(son_okumalar):,}")
    
    with col2:
        toplam_tuketim = son_okumalar['AKTIF_m3'].sum() if 'AKTIF_m3' in son_okumalar.columns else 0
        st.metric("💧 Toplam Tüketim", f"{toplam_tuketim:,.0f} m³")
    
    with col3:
        toplam_gelir = son_okumalar['TOPLAM_TUTAR'].sum() if 'TOPLAM_TUTAR' in son_okumalar.columns else 0
        st.metric("💰 Toplam Gelir", f"{toplam_gelir:,.0f} TL")
    
    with col4:
        if 'RISK_SEVIYESI' in son_okumalar.columns:
            yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        else:
            yuksek_riskli = 0
        st.metric("🚨 Yüksek Riskli", f"{yuksek_riskli}")
    
    with col5:
        st.metric("🧠 AI Gözlem", f"{learning_stats['toplam_gozlem']:,}")

# Tab Menü - GÜNCELLENMİŞ
tab1, tab2, tab3, tab4 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz", "🤖 AI Öğrenme"])

with tab1:
    if son_okumalar is not None and len(son_okumalar) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                                  title='Gerçekçi Günlük Tüketim Dağılımı',
                                  labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'})
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
                fig2 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                             title='Öğrenilmiş Risk Dağılımı')
                st.plotly_chart(fig2, use_container_width=True)

with tab2:
    if zone_analizi is not None and len(zone_analizi) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                        title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Zone Karşılaştırma")
            zone_gosterim = zone_analizi.copy()
            if 'TOPLAM_TUKETIM' in zone_gosterim.columns:
                zone_gosterim['TOPLAM_TUKETIM'] = zone_gosterim['TOPLAM_TUKETIM'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
            if 'TOPLAM_GELIR' in zone_gosterim.columns:
                zone_gosterim['TOPLAM_GELIR'] = zone_gosterim['TOPLAM_GELIR'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
            st.dataframe(zone_gosterim, use_container_width=True)

with tab3:
    if son_okumalar is not None and len(son_okumalar) > 0:
        st.subheader("Öğrenilmiş Tesisat Detayları")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_filtre = st.multiselect(
                    "Risk Seviyesi",
                    options=son_okumalar['RISK_SEVIYESI'].unique(),
                    default=son_okumalar['RISK_SEVIYESI'].unique()
                )
        
        with col2:
            siralama = st.selectbox("Sıralama", ["Tüketim (Azalan)", "Tüketim (Artan)", "Risk Puanı"])
        
        filtreli_veri = son_okumalar
        if risk_filtre and 'RISK_SEVIYESI' in son_okumalar.columns:
            filtreli_veri = filtreli_veri[filtreli_veri['RISK_SEVIYESI'].isin(risk_filtre)]
        
        if siralama == "Tüketim (Azalan)":
            filtreli_veri = filtreli_veri.sort_values('AKTIF_m3', ascending=False)
        elif siralama == "Tüketim (Artan)":
            filtreli_veri = filtreli_veri.sort_values('AKTIF_m3', ascending=True)
        elif siralama == "Risk Puanı" and 'RISK_PUANI' in filtreli_veri.columns:
            filtreli_veri = filtreli_veri.sort_values('RISK_PUANI', ascending=False)
        
        def format_tesisat_no(tesisat_no):
            if pd.isna(tesisat_no):
                return ""
            cleaned = str(tesisat_no).strip()
            digits_only = re.sub(r'\D', '', cleaned)
            return digits_only
        
        gosterilecek_veri = filtreli_veri.copy()
        gosterilecek_veri['TESISAT_NO'] = gosterilecek_veri['TESISAT_NO'].apply(format_tesisat_no)
        
        gosterilecek_kolonlar = ['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR']
        if 'GUNLUK_ORT_TUKETIM_m3' in gosterilecek_veri.columns:
            gosterilecek_kolonlar.append('GUNLUK_ORT_TUKETIM_m3')
        if 'RISK_SEVIYESI' in gosterilecek_veri.columns:
            gosterilecek_kolonlar.append('RISK_SEVIYESI')
        if 'DAVRANIS_YORUMU' in gosterilecek_veri.columns:
            gosterilecek_kolonlar.append('DAVRANIS_YORUMU')
        
        def format_numeric_columns(df):
            formatted_df = df.copy()
            if 'AKTIF_m3' in formatted_df.columns:
                formatted_df['AKTIF_m3'] = formatted_df['AKTIF_m3'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            if 'TOPLAM_TUTAR' in formatted_df.columns:
                formatted_df['TOPLAM_TUTAR'] = formatted_df['TOPLAM_TUTAR'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            if 'GUNLUK_ORT_TUKETIM_m3' in formatted_df.columns:
                formatted_df['GUNLUK_ORT_TUKETIM_m3'] = formatted_df['GUNLUK_ORT_TUKETIM_m3'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            return formatted_df
        
        gosterilecek_veri_formatted = format_numeric_columns(gosterilecek_veri)
        st.dataframe(gosterilecek_veri_formatted[gosterilecek_kolonlar].head(50), use_container_width=True)

with tab4:
    st.header("🤖 Adaptive AI Öğrenme Durumu")
    
    # Detaylı öğrenme istatistikleri
    stats = adaptive_model.get_learning_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👁️ Toplam Gözlem", f"{stats['toplam_gozlem']:,}")
    with col2:
        st.metric("🎯 Gerçek Gözlem", f"{stats['gercek_gozlem']:,}")
    with col3:
        st.metric("✅ Başarı Oranı", f"{stats['basari_orani']:.1%}")
    with col4:
        st.metric("🔢 Model Versiyon", stats['model_version'])
    
    # Öğrenme durumu - DÜZELTİLMİŞ
    st.subheader("📊 Öğrenme İlerlemesi")
    
    progress_col1, progress_col2 = st.columns(2)
    
    with progress_col1:
        # Gözlem ilerlemesi - DÜZELTİLMİŞ
        total_obs = stats['toplam_gozlem']
        max_obs = 10000
        obs_progress = min(total_obs / max_obs, 1.0)
        st.progress(obs_progress)
        st.write(f"**Gözlem İlerlemesi:** {total_obs:,} / {max_obs:,}")
    
    with progress_col2:
        # Başarı ilerlemesi - DÜZELTİLMİŞ
        success_rate = stats['basari_orani']
        st.progress(success_rate)
        st.write(f"**Başarı Oranı:** {success_rate:.1%}")
    
    # Adaptive threshold grafiği - BASİTLEŞTİRİLMİŞ
    st.subheader("🔧 Adaptive Threshold Değerleri")
    
    thresholds = stats['adaptive_thresholds']
    
    # Tablo formatında göster
    threshold_data = []
    for key, value in thresholds.items():
        threshold_data.append({
            'Threshold': key,
            'Değer': f"{value:.2f}"
        })
    
    threshold_df = pd.DataFrame(threshold_data)
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)
    
    # Basit grafik
    fig = px.bar(threshold_df, x='Threshold', y='Değer', 
                 title='Adaptive Threshold Değerleri',
                 labels={'Değer': 'Threshold Değeri'})
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Önerileri - NETLEŞTİRİLMİŞ
    st.subheader("🚀 AI Önerileri & Sonraki Adımlar")
    
    if stats['toplam_gozlem'] < 50:
        st.info("""
        **🎯 Öneri:** AI henüz yeni başladı! 
        - Demo modda birkaç analiz yapın
        - Hızlı geri bildirim butonlarını kullanın  
        - 50+ gözlem sonrası daha akıllı hale gelecek
        """)
    elif stats['basari_orani'] < 0.7:
        st.warning("""
        **⚠️ Geliştirme Gerekli:** Başarı oranı düşük!
        - Daha fazla geri bildirim toplayın
        - Threshold'ları manuel ayarlamayı düşünün
        - Farklı pattern'ler için feedback verin
        """)
    else:
        st.success("""
        **✅ Mükemmel Performans:** AI iyi öğreniyor!
        - Mevcut ayarları koruyun
        - Yeni pattern'ler için feedback vermeye devam edin
        - Modeli düzenli olarak kaydedin
        """)
    
    # Pattern hafızası - NET GÖSTERİM
    if stats['pattern_memory_size'] > 0:
        st.subheader("🧠 Pattern Hafızası")
        st.info(f"AI **{stats['pattern_memory_size']}** farklı pattern'i hafızasında tutuyor!")
    
    # EK: Model durumu
    st.subheader("🔍 Model Durumu")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.write(f"**Durum:** {stats['status']}")
        st.write(f"**Son Güncelleme:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    with status_col2:
        st.write(f"**Gerçek Başarı Oranı:** {stats.get('gercek_basari_orani', 0):.1%}")
        st.write(f"**Pattern Bellek Kullanımı:** {stats['pattern_memory_size']} / 5,000")

# Footer
st.markdown("---")
st.markdown("🚀 **Hemen Öğrenen Su Tüketim AI Sistemi v2.0** | 1M+ Satır Desteği | Optimize Bellek Yönetimi")
