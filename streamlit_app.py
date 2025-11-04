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
# 🧠 ADAPTIVE STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Adaptive Su Tüketim AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ======================================================================
# 📊 AKILLI VERİ İŞLEME FONKSİYONLARI
# ======================================================================

@st.cache_data(ttl=3600)  # 1 saat cache
def load_and_analyze_data_adaptive(uploaded_file, zone_file):
    """Adaptive analiz ile veri işleme"""
    try:
        # Ana veri dosyasını oku - tesisat no'yu string olarak oku
        df = pd.read_excel(uploaded_file, dtype={'TESISAT_NO': str})
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
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

    # Tesisat numaralarını temizle
    df['TESISAT_NO'] = df['TESISAT_NO'].apply(clean_tesisat_no)
    df = df[df['TESISAT_NO'].notnull()]
    df = df[df['TESISAT_NO'] != '']
    df = df[df['TESISAT_NO'] != 'nan']
    
    # Tarih formatını düzelt
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
    
    # Sayısal sütunları temizle
    numeric_columns = ['AKTIF_m3', 'TOPLAM_TUTAR']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Temel özellik mühendisliği
    df['OKUMA_PERIYODU_GUN'] = (df['OKUMA_TARIHI'] - df['ILK_OKUMA_TARIHI']).dt.days
    df['OKUMA_PERIYODU_GUN'] = df['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
    df['GUNLUK_ORT_TUKETIM_m3'] = df['AKTIF_m3'] / df['OKUMA_PERIYODU_GUN']
    df['GUNLUK_ORT_TUKETIM_m3'] = df['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
    
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

    # ADAPTIVE analiz yap
    if len(son_okumalar) > 0:
        st.info("🧠 Adaptive AI analizi yapılıyor...")
        progress_bar = st.progress(0)
        davranis_sonuclari = []
        
        total_tesisat = len(son_okumalar)
        for i, (idx, row) in enumerate(son_okumalar.iterrows()):
            tesisat_verisi = df[df['TESISAT_NO'] == row['TESISAT_NO']].sort_values('OKUMA_TARIHI')
            
            # Adaptive analiz
            analiz_sonucu = adaptive_model.gelismis_davranis_analizi(tesisat_verisi)
            
            davranis_sonuclari.append({
                'TESISAT_NO': row['TESISAT_NO'],
                'DAVRANIS_YORUMU': analiz_sonucu['yorum'],
                'SUPHELI_DONEMLER': analiz_sonucu['supheli_donemler'],
                'RISK_SEVIYESI': analiz_sonucu['risk_seviyesi'],
                'RISK_PUANI': analiz_sonucu['risk_puan']
            })
            
            if i % 100 == 0 and total_tesisat > 0:
                progress_bar.progress(min((i + 1) / total_tesisat, 1.0))

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
# 🎨 ADAPTIVE STREAMLIT ARAYÜZ
# ======================================================================

st.title("🧠 Adaptive Su Tüketim AI Dashboard")
st.markdown("**Kendi Kendine Öğrenen Yapay Zeka Sistemi**")

# Sidebar - Öğrenme Kontrolleri
st.sidebar.header("🧠 AI Öğrenme Kontrolleri")

# Öğrenme istatistikleri
learning_stats = adaptive_model.get_learning_stats()
st.sidebar.metric("🤖 Toplam Gözlem", f"{learning_stats['toplam_gozlem']:,}")
st.sidebar.metric("🎯 Başarı Oranı", f"{learning_stats['basari_orani']:.1%}")

# Adaptive threshold'ları göster
st.sidebar.subheader("Adaptive Threshold'lar")
for key, value in learning_stats['adaptive_thresholds'].items():
    st.sidebar.write(f"**{key}**: {value:.2f}")

# Geri bildirim sistemi
st.sidebar.subheader("🤖 AI Geri Bildirim")
feedback_tesisat = st.sidebar.text_input("Tesisat No")
feedback_gercek = st.sidebar.selectbox("Gerçek Durum", ["Yüksek", "Orta", "Düşük"])
feedback_tahmin = st.sidebar.selectbox("Tahmin Durum", ["Yüksek", "Orta", "Düşük"])

if st.sidebar.button("📝 Geri Bildirim Gönder"):
    if feedback_tesisat:
        adaptive_model.learn_from_feedback(feedback_tesisat, feedback_gercek, feedback_tahmin)
        st.sidebar.success("✅ Geri bildirim kaydedildi! AI öğreniyor...")
        st.rerun()

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

# Demo butonu - Adaptive
if st.sidebar.button("🎮 Adaptive Demo Modu"):
    st.info("🧠 Adaptive Demo modu aktif! AI öğrenme mekanizması çalışıyor...")
    
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
    
    # Temel özellik mühendisliği
    df_detayli['OKUMA_PERIYODU_GUN'] = 300
    df_detayli['GUNLUK_ORT_TUKETIM_m3'] = df_detayli['AKTIF_m3'] / df_detayli['OKUMA_PERIYODU_GUN']
    
    # Son okumaları al
    son_okumalar = df_detayli.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
    
    # Adaptive analiz
    davranis_sonuclari = []
    for tesisat_no in son_okumalar['TESISAT_NO'].unique():
        tesisat_verisi = df_detayli[df_detayli['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
        analiz_sonucu = adaptive_model.gelismis_davranis_analizi(tesisat_verisi)
        
        davranis_sonuclari.append({
            'TESISAT_NO': tesisat_no,
            'DAVRANIS_YORUMU': analiz_sonucu['yorum'],
            'SUPHELI_DONEMLER': analiz_sonucu['supheli_donemler'],
            'RISK_SEVIYESI': analiz_sonucu['risk_seviyesi'],
            'RISK_PUANI': analiz_sonucu['risk_puan']
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
    
    st.success("✅ Adaptive demo verisi başarıyla oluşturuldu! AI öğrenme aktif.")

elif uploaded_file is not None:
    try:
        df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data_adaptive(uploaded_file, zone_file)
        if df is None:
            st.stop()
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        st.stop()
else:
    st.warning("⚠️ Lütfen Excel dosyasını yükleyin veya Adaptive Demo modunu kullanın")
    st.stop()

# Genel Metrikler - Adaptive
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

# Tab Menü - Adaptive
tab1, tab2, tab3, tab4 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz", "🤖 AI Öğrenme"])

with tab1:
    if son_okumalar is not None and len(son_okumalar) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                                  title='Adaptive Günlük Tüketim Dağılımı',
                                  labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'})
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
                fig2 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                             title='Adaptive Risk Dağılımı')
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
        st.subheader("Adaptive Tesisat Detayları")
        
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
    
    # Öğrenme istatistikleri
    stats = adaptive_model.get_learning_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam Gözlem", f"{stats['toplam_gozlem']:,}")
    with col2:
        st.metric("Başarı Oranı", f"{stats['basari_orani']:.1%}")
    with col3:
        st.metric("Model Versiyon", "1.0")
    
    # Adaptive threshold grafiği
    st.subheader("Adaptive Threshold Gelişimi")
    thresholds_df = pd.DataFrame([stats['adaptive_thresholds']])
    st.dataframe(thresholds_df.T.rename(columns={0: 'Değer'}), use_container_width=True)
    
    # Öğrenme önerileri
    st.subheader("🤖 AI Önerileri")
    if stats['toplam_gozlem'] < 100:
        st.info("**Öneri**: Daha fazla geri bildirim toplayarak AI'nın öğrenme performansını artırabilirsiniz.")
    elif stats['basari_orani'] < 0.7:
        st.warning("**Öneri**: Başarı oranı düşük. Threshold değerlerini manuel olarak ayarlamayı düşünebilirsiniz.")
    else:
        st.success("**Öneri**: AI iyi performans gösteriyor! Mevcut ayarları koruyabilirsiniz.")

# Footer
st.markdown("---")
st.markdown("🧠 **Adaptive Su Tüketim AI Sistemi** | Kendi Kendine Öğrenen Yapay Zeka")
