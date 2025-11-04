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

# Model import
try:
    from model_api import analiz_modeli
except ImportError:
    st.error("Model API yüklenemedi. Lütfen model_api.py dosyasını kontrol edin.")
    st.stop()

# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Su Tüketim Davranış Analiz Dashboard",
    page_icon="💧",
    layout="wide"
)

# ======================================================================
# 📊 VERİ İŞLEME FONKSİYONLARI
# ======================================================================

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder"""
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
        # String'e çevir ve temizle
        tesisat_str = str(tesisat_no).strip()
        # Noktalama işaretlerini kaldır
        tesisat_str = re.sub(r'[,"\']', '', tesisat_str)
        # Baştaki ve sondaki boşlukları temizle
        tesisat_str = tesisat_str.strip()
        return tesisat_str

    # Tesisat numaralarını temizle
    df['TESISAT_NO'] = df['TESISAT_NO'].apply(clean_tesisat_no)
    
    # Tesisat numarası olan kayıtları filtrele
    df = df[df['TESISAT_NO'].notnull()]
    df = df[df['TESISAT_NO'] != '']
    df = df[df['TESISAT_NO'] != 'nan']
    
    # Tarih formatını düzelt
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
    
    # Model ile veri ön işleme
    df = analiz_modeli.veri_on_isleme(df)
    
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
            st.warning(f"⚠️ Zone veri dosyası yüklenirken hata: {e}")

    # Son okumaları al
    son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()

    # Tüm tesisatlar için davranış analizi yap
    if len(son_okumalar) > 0:
        st.info("🔍 Davranış analizi yapılıyor...")
        progress_bar = st.progress(0)
        davranis_sonuclari = []
        
        total_tesisat = len(son_okumalar)
        for i, (idx, row) in enumerate(son_okumalar.iterrows()):
            tesisat_verisi = df[df['TESISAT_NO'] == row['TESISAT_NO']].sort_values('OKUMA_TARIHI')
            yorum, supheli_donemler, risk, risk_puan = analiz_modeli.gelismis_davranis_analizi(tesisat_verisi)
            
            davranis_sonuclari.append({
                'TESISAT_NO': row['TESISAT_NO'],  # Temizlenmiş tesisat no
                'DAVRANIS_YORUMU': yorum,
                'SUPHELI_DONEMLER': supheli_donemler,
                'RISK_SEVIYESI': risk,
                'RISK_PUANI': risk_puan
            })
            
            # Progress bar güncelleme
            if i % 100 == 0 and total_tesisat > 0:
                progress_bar.progress(min((i + 1) / total_tesisat, 1.0))

        progress_bar.progress(1.0)
        davranis_df = pd.DataFrame(davranis_sonuclari)
        son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

        # Anomali tespiti
        son_okumalar = analiz_modeli.anomaly_detection(son_okumalar)

    # Zone analizi
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        # Son 3 aylık veriyi al
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

        # Zone risk analizi
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

        # Kullanıcı zone verilerini birleştir
        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi, kullanici_zone_verileri

def create_demo_data():
    """Demo verisi oluşturur"""
    np.random.seed(42)
    
    # Basit demo verisi oluştur
    demo_data = []
    tesisat_sayisi = 500  # Daha küçük demo verisi
    
    for i in range(tesisat_sayisi):
        tesisat_no = f"8000{300 + i}"  # Temiz tesisat numaraları
        
        # Basit tüketim patternleri
        pattern_type = np.random.choice(['normal', 'sifir_aralikli', 'yuksek'], p=[0.7, 0.2, 0.1])
        
        if pattern_type == 'normal':
            aktif_m3 = np.random.gamma(2, 8)
        elif pattern_type == 'sifir_aralikli':
            aktif_m3 = 0 if np.random.random() < 0.3 else np.random.gamma(2, 6)
        else:  # yuksek
            aktif_m3 = np.random.gamma(5, 15)
        
        toplam_tutar = aktif_m3 * 15
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Çoklu okuma verisi oluştur
    coklu_okuma_data = []
    for tesisat in demo_data:
        tesisat_no = tesisat['TESISAT_NO']
        base_consumption = tesisat['AKTIF_m3']
        
        # 3 aylık veri oluştur
        for month in range(3):
            month_date = pd.Timestamp(f'2024-{8+month:02d}-15')
            
            if 'sifir_aralikli' in tesisat_no and month == 1:
                consumption = 0
            elif 'yuksek' in tesisat_no:
                consumption = base_consumption * (1 + np.random.normal(0, 0.5))
            else:
                consumption = base_consumption * (1 + np.random.normal(0, 0.2))
            
            coklu_okuma_data.append({
                'TESISAT_NO': tesisat_no,
                'AKTIF_m3': max(consumption, 0),
                'TOPLAM_TUTAR': max(consumption * 15, 0),
                'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
                'OKUMA_TARIHI': month_date,
                'KARNE_NO': tesisat['KARNE_NO']
            })
    
    return pd.DataFrame(coklu_okuma_data)

# ======================================================================
# 🎨 STREAMLIT ARAYÜZ
# ======================================================================

# Başlık
st.title("💧 Su Tüketim Davranış Analiz Dashboard")
st.markdown("**Gelişmiş Analiz Sistemi**")

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

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    
    try:
        df_detayli = create_demo_data()
        df_detayli = analiz_modeli.veri_on_isleme(df_detayli)
        
        # Son okumaları al
        son_okumalar = df_detayli.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        
        # Demo için analiz yap
        davranis_sonuclari = []
        for tesisat_no in son_okumalar['TESISAT_NO'].unique():
            tesisat_verisi = df_detayli[df_detayli['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
            yorum, donemler, risk, risk_puan = analiz_modeli.gelismis_davranis_analizi(tesisat_verisi)
            
            davranis_sonuclari.append({
                'TESISAT_NO': tesisat_no,
                'DAVRANIS_YORUMU': yorum,
                'SUPHELI_DONEMLER': donemler,
                'RISK_SEVIYESI': risk,
                'RISK_PUANI': risk_puan
            })
        
        davranis_df = pd.DataFrame(davranis_sonuclari)
        son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')
        
        # Anomali tespiti
        son_okumalar = analiz_modeli.anomaly_detection(son_okumalar)
        
        # Zone analizi
        zone_analizi = df_detayli.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
        
        # Risk analizi zone bazlı
        zone_risk = son_okumalar.groupby('KARNE_NO')['RISK_SEVIYESI'].apply(
            lambda x: (x == 'Yüksek').sum()
        ).reset_index(name='YUKSEK_RISKLI_TESISAT')
        
        zone_analizi = zone_analizi.merge(zone_risk, on='KARNE_NO', how='left')
        zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100
        
        # Örnek zone verileri
        kullanici_zone_verileri = {
            'ZONE1': {'ad': 'BÖLGE-1', 'verilen_su': 10000, 'tahakkuk_m3': 7000, 'kayip_oran': 30.0},
            'ZONE2': {'ad': 'BÖLGE-2', 'verilen_su': 8000, 'tahakkuk_m3': 6000, 'kayip_oran': 25.0},
        }
        
        st.success("✅ Demo verisi başarıyla oluşturuldu!")
        df = df_detayli
        
    except Exception as e:
        st.error(f"Demo verisi oluşturulurken hata: {e}")
        st.stop()

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    try:
        df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
        if df is None:
            st.stop()
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        st.stop()
else:
    st.warning("⚠️ Lütfen Excel dosyasını yükleyin veya Demo modunu kullanın")
    st.stop()

# Genel Metrikler
if son_okumalar is not None and len(son_okumalar) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
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

# Tab Menü
tab1, tab2, tab3 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz"])

with tab1:
    if son_okumalar is not None and len(son_okumalar) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tüketim Dağılım Grafiği
            if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                                  title='Günlük Tüketim Dağılımı',
                                  labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'})
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Risk Dağılımı
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
                fig2 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                             title='Risk Seviyeleri Dağılımı')
                st.plotly_chart(fig2, use_container_width=True)

with tab2:
    if zone_analizi is not None and len(zone_analizi) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone Tüketim Dağılımı
            fig3 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                        title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Zone Karşılaştırma
            st.subheader("Zone Karşılaştırma")
            
            # Zone verisini formatla
            zone_gosterim = zone_analizi.copy()
            if 'TOPLAM_TUKETIM' in zone_gosterim.columns:
                zone_gosterim['TOPLAM_TUKETIM'] = zone_gosterim['TOPLAM_TUKETIM'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
            if 'TOPLAM_GELIR' in zone_gosterim.columns:
                zone_gosterim['TOPLAM_GELIR'] = zone_gosterim['TOPLAM_GELIR'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
            if 'YUKSEK_RISK_ORANI' in zone_gosterim.columns:
                zone_gosterim['YUKSEK_RISK_ORANI'] = zone_gosterim['YUKSEK_RISK_ORANI'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0%")
            
            st.dataframe(zone_gosterim, use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    if son_okumalar is not None and len(son_okumalar) > 0:
        st.subheader("Tesisat Detayları")
        
        # Filtreleme
        col1, col2 = st.columns(2)
        with col1:
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_filtre = st.multiselect(
                    "Risk Seviyesi",
                    options=son_okumalar['RISK_SEVIYESI'].unique(),
                    default=son_okumalar['RISK_SEVIYESI'].unique()
                )
            else:
                risk_filtre = []
        
        with col2:
            siralama = st.selectbox("Sıralama", ["Tüketim (Azalan)", "Tüketim (Artan)", "Risk Puanı"])
        
        # Filtreleme uygula
        filtreli_veri = son_okumalar
        if risk_filtre and 'RISK_SEVIYESI' in son_okumalar.columns:
            filtreli_veri = filtreli_veri[filtreli_veri['RISK_SEVIYESI'].isin(risk_filtre)]
        
        # Sıralama uygula
        if siralama == "Tüketim (Azalan)":
            filtreli_veri = filtreli_veri.sort_values('AKTIF_m3', ascending=False)
        elif siralama == "Tüketim (Artan)":
            filtreli_veri = filtreli_veri.sort_values('AKTIF_m3', ascending=True)
        elif siralama == "Risk Puanı" and 'RISK_PUANI' in filtreli_veri.columns:
            filtreli_veri = filtreli_veri.sort_values('RISK_PUANI', ascending=False)
        
        # Tablo gösterimi - tesisat no formatını düzelt
        gosterilecek_veri = filtreli_veri.copy()
        
        # Tesisat numarasını temizle ve formatla
        def format_tesisat_no(tesisat_no):
            if pd.isna(tesisat_no):
                return ""
            # String'e çevir ve temizle
            cleaned = str(tesisat_no).strip()
            # Sadece rakamları al
            digits_only = re.sub(r'\D', '', cleaned)
            return digits_only
        
        gosterilecek_veri['TESISAT_NO'] = gosterilecek_veri['TESISAT_NO'].apply(format_tesisat_no)
        
        gosterilecek_kolonlar = ['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR']
        if 'GUNLUK_ORT_TUKETIM_m3' in gosterilecek_veri.columns:
            gosterilecek_kolonlar.append('GUNLUK_ORT_TUKETIM_m3')
        if 'RISK_SEVIYESI' in gosterilecek_veri.columns:
            gosterilecek_kolonlar.append('RISK_SEVIYESI')
        if 'DAVRANIS_YORUMU' in gosterilecek_veri.columns:
            gosterilecek_kolonlar.append('DAVRANIS_YORUMU')
        
        # Sayısal sütunları formatla
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

# Footer
st.markdown("---")
st.markdown("💧 **Su Tüketim Analiz Sistemi** | Streamlit Dashboard")
