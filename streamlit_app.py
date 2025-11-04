# streamlit_app.py - GÜNCELLENMİŞ VERSİYON
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import time
import re
import subprocess
import sys


# ======================================================================
# API CLIENT - AYNI
# ======================================================================
class ModelAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "SuTuketimAI-Streamlit/1.0"
        })
    
    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def incremental_learn(self, data: list, batch_id: str = None) -> dict:
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
        
        try:
            response = self.session.post(
                f"{self.base_url}/incremental-learn",
                json={
                    "data": data,
                    "batch_id": batch_id
                },
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict(self, data: dict) -> dict:
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=10
            )
            return response.json()
        except:
            return {"score": 0.0, "risk_level": "Bilinmiyor"}
    
    def get_model_info(self) -> dict:
        try:
            response = self.session.get(f"{self.base_url}/model-info", timeout=5)
            return response.json()
        except:
            return {"status": "error"}

# ======================================================================
# VERİ İŞLEME FONKSİYONLARI - İKİ DOSYA İÇİN
# ======================================================================
@st.cache_data(ttl=3600)
def load_and_analyze_data(uploaded_file, zone_file):
    """İKİ DOSYADAN veriyi okur ve analiz eder"""
    try:
        # 1. DOSYA: Ana veri (yavuz.xlsx)
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None

    # Tarih formatını düzelt
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    
    # Tesisat numarası olan kayıtları filtrele
    df = df[df['TESISAT_NO'].notnull()]
    
    # 2. DOSYA: Zone veri dosyasını oku (YAVUZELİ MERKEZ 2025 EKİM)
    kullanici_zone_verileri = {}
    zone_excel_df = None
    
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

    # Davranış analizi fonksiyonu
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Basit risk analizi fonksiyonu
    def quick_risk_analysis(tesisat_no, df):
        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
        
        if len(tesisat_verisi) < 2:
            return "Yetersiz veri", "Düşük"

        tuketimler = tesisat_verisi['AKTIF_m3'].values
        sifir_sayisi = sum(tuketimler == 0)
        son_tuketim = tuketimler[-1] if len(tuketimler) > 0 else 0
        
        if sifir_sayisi >= 2 or son_tuketim == 0:
            return "Düzensiz tüketim", "Yüksek"
        elif sifir_sayisi >= 1:
            return "Ara sıra sıfır", "Orta"
        else:
            return "Normal patern", "Düşük"

    # Tüm tesisatlar için analiz
    davranis_sonuclari = []
    for idx, row in son_okumalar.iterrows():
        yorum, risk = quick_risk_analysis(row['TESISAT_NO'], df)
        davranis_sonuclari.append({
            'TESISAT_NO': row['TESISAT_NO'],
            'DAVRANIS_YORUMU': yorum,
            'RISK_SEVIYESI': risk
        })

    davranis_df = pd.DataFrame(davranis_sonuclari)
    son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    # Zone analizi - EKİM 2024 verisi
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

        # Kullanıcı zone verilerini birleştir
        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi

# ======================================================================
# STREAMLIT APP - İKİ DOSYA YÜKLEMELİ
# ======================================================================
st.set_page_config(
    page_title="🤖 Su Tüketim AI - Profesyonel",
    page_icon="💧",
    layout="wide"
)

# API Client initialization
API_URL = st.secrets.get("API_URL", "http://localhost:8000")
api_client = ModelAPIClient(API_URL)

st.title("💧 Su Tüketim AI Analiz Sistemi")
st.markdown("🚀 **Profesyonel Mimari: FastAPI + Streamlit + Incremental Learning**")

# ======================================================================
# SIDEBAR - API & MODEL YÖNETİMİ
# ======================================================================
st.sidebar.header("🔗 API Bağlantı")

# API durumu
if api_client.health_check():
    st.sidebar.success("✅ API Bağlantısı Aktif")
    
    # Model bilgileri
    model_info = api_client.get_model_info()
    if model_info.get("status") != "error":
        st.sidebar.metric("🤖 Model", model_info.get("model_type", "River"))
        st.sidebar.metric("📚 İşlenen Veri", f"{model_info.get('stats', {}).get('total_processed', 0):,}")
else:
    st.sidebar.error("❌ API Bağlantısı Yok")
    st.sidebar.info("🔧 Backend API henüz hazır değil")

# ======================================================================
# İKİ DOSYA YÜKLEME - GÜNCELLENMİŞ
# ======================================================================
st.sidebar.header("📁 Çift Dosya Yükleme")

st.sidebar.markdown("**1. Ana Veri Dosyası**")
uploaded_file = st.sidebar.file_uploader(
    "yavuz.xlsx dosyasını seçin",
    type=["xlsx"],
    help="Tüm tesisat verilerini içeren ana Excel dosyası"
)

st.sidebar.markdown("**2. Zone Veri Dosyası**")
zone_file = st.sidebar.file_uploader(
    "YAVUZELİ MERKEZ 2025 EKİM.xlsx dosyasını seçin", 
    type=["xlsx"],
    help="Zone bazlı özet verileri içeren Excel dosyası"
)

# Learning ayarları
st.sidebar.header("🎯 AI Öğrenme Kontrol")
auto_learn = st.sidebar.checkbox("🔄 Otomatik Incremental Learning", value=True)
batch_size = st.sidebar.slider("📦 Batch Boyutu", 100, 2000, 500)

# ======================================================================
# ANA UYGULAMA LOGIC - İKİ DOSYA İLE
# ======================================================================
if uploaded_file is not None:
    # İKİ DOSYA ile veriyi yükle
    with st.spinner("📊 İki dosyadan veri yükleniyor ve analiz ediliyor..."):
        df, son_okumalar, zone_analizi = load_and_analyze_data(uploaded_file, zone_file)
    
    if df is not None and son_okumalar is not None:
        st.success(f"✅ {len(df)} kayıt yüklendi | {len(son_okumalar)} tesisat analiz edildi")
        
        # Zone dosyası kontrolü
        if zone_file is None:
            st.warning("⚠️ Zone dosyası yüklenmedi - zone analizi sınırlı")
        else:
            st.success("🗺️ Zone analizi için veriler yüklendi")
        
        # INCREMENTAL LEARNING - API üzerinden
        if auto_learn and api_client.health_check():
            with st.spinner("🤖 AI yeni veriyi öğreniyor..."):
                # Sadece ana veriden learning yap
                batch_data = df.head(batch_size).to_dict('records')
                learn_result = api_client.incremental_learn(batch_data)
                
                if learn_result.get("status") == "success":
                    st.success(f"🎯 {learn_result['processed']} kayıt öğrenildi")
                else:
                    st.error(f"❌ Öğrenme hatası: {learn_result.get('message', 'Bilinmeyen hata')}")
        
        # ======================================================================
        # ANALIZ VE GÖRSELLEŞTİRME
        # ======================================================================
        
        # AI Tahminleri al
        if api_client.health_check():
            with st.spinner("🔮 AI tahminleri hesaplanıyor..."):
                ai_scores = []
                ai_risks = []
                
                for _, row in son_okumalar.iterrows():
                    prediction = api_client.predict(row.to_dict())
                    ai_scores.append(prediction.get('score', 0))
                    ai_risks.append(prediction.get('risk_level', 'Bilinmiyor'))
                
                son_okumalar['AI_SKOR'] = ai_scores
                son_okumalar['AI_RISK'] = ai_risks
        
        # METRIKLER
        st.header("📊 Gerçek Zamanlı Metrikler")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏠 Toplam Tesisat", f"{len(son_okumalar):,}")
        
        with col2:
            st.metric("💧 Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")
        
        with col3:
            geleneksel_yuksek = (son_okumalar['RISK_SEVIYESI'] == 'Yüksek').sum()
            st.metric("🎯 Geleneksel Risk", geleneksel_yuksek)
        
        with col4:
            if 'AI_RISK' in son_okumalar.columns:
                ai_yuksek = (son_okumalar['AI_RISK'] == 'Yüksek').sum()
                st.metric("🤖 AI Risk", ai_yuksek)
            else:
                st.metric("🤖 AI", "Pasif")
        
        # GÖRSELLEŞTİRMELER
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Temel Analiz", "🗺️ Zone Analiz", "🤖 AI Insights", "🚨 Riskler"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                                  title='Günlük Tüketim Dağılımı')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                                color='RISK_SEVIYESI',
                                title='Tüketim-Tutar İlişkisi',
                                color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            if zone_analizi is not None:
                col1, col2 = st.columns(2)
                with col1:
                    fig3 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                                 title='Zone Bazlı Tüketim Dağılımı')
                    st.plotly_chart(fig3, use_container_width=True)
                with col2:
                    fig4 = px.bar(zone_analizi, x='KARNE_NO', y='TESISAT_SAYISI',
                                 title='Zone Bazlı Tesisat Sayısı',
                                 color_discrete_sequence=['#E74C3C'])
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Zone karşılaştırma tablosu
                st.subheader("Zone Karşılaştırma Tablosu")
                zone_karsilastirma = zone_analizi[['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']].copy()
                if 'ad' in zone_analizi.columns:
                    zone_karsilastirma['Zone Adı'] = zone_analizi['ad']
                if 'verilen_su' in zone_analizi.columns:
                    zone_karsilastirma['Verilen Su (m³)'] = zone_analizi['verilen_su']
                    zone_karsilastirma['Kayıp Oranı (%)'] = zone_analizi['kayip_oran']
                
                st.dataframe(zone_karsilastirma, use_container_width=True)
            else:
                st.info("Zone analizi için veri bulunamadı")
        
        with tab3:
            if 'AI_SKOR' in son_okumalar.columns:
                col1, col2 = st.columns(2)
                with col1:
                    fig5 = px.histogram(son_okumalar, x='AI_SKOR', 
                                      title='AI Anomali Skor Dağılımı',
                                      nbins=30,
                                      color_discrete_sequence=['#FF6B6B'])
                    st.plotly_chart(fig5, use_container_width=True)
                with col2:
                    # En yüksek AI riskliler
                    high_ai_risk = son_okumalar[son_okumalar['AI_RISK'] == 'Yüksek']
                    if len(high_ai_risk) > 0:
                        st.dataframe(high_ai_risk[
                            ['TESISAT_NO', 'AI_SKOR', 'AKTIF_m3', 'RISK_SEVIYESI']
                        ].head(10), use_container_width=True)
                    else:
                        st.success("🎉 AI yüksek risk bulamadı!")
            else:
                st.info("🤖 AI analiz için API bağlantısı gerekli")
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                geleneksel_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
                fig6 = px.pie(values=geleneksel_dagilim.values, 
                            names=geleneksel_dagilim.index,
                            title='Geleneksel Risk Dağılımı',
                            color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
                st.plotly_chart(fig6, use_container_width=True)
            
            with col2:
                if 'AI_RISK' in son_okumalar.columns:
                    ai_dagilim = son_okumalar['AI_RISK'].value_counts()
                    fig7 = px.pie(values=ai_dagilim.values, 
                                names=ai_dagilim.index,
                                title='AI Risk Dağılımı',
                                color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
                    st.plotly_chart(fig7, use_container_width=True)
                else:
                    st.info("AI risk dağılımı için API gerekli")
            
            # Yüksek riskli tesisatlar
            st.subheader("🚨 Yüksek Riskli Tesisatlar")
            high_risk_tesisatlar = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
            if len(high_risk_tesisatlar) > 0:
                st.dataframe(high_risk_tesisatlar[
                    ['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'DAVRANIS_YORUMU']
                ].head(15), use_container_width=True)
            else:
                st.success("🎉 Yüksek riskli tesisat bulunamadı!")

else:
    # LANDING PAGE
    st.info("👆 Lütfen İKİ Excel dosyasını da yükleyin")
    
    st.markdown("""
    **📁 Gerekli Dosyalar:**
    1. **yavuz.xlsx** - Tüm tesisat verileri
    2. **YAVUZELİ MERKEZ 2025 EKİM.xlsx** - Zone bazlı özet veriler
    
    **🔧 Sistem Özellikleri:**
    - İki dosyadan entegre analiz
    - Geleneksel risk analizi 
    - AI destekli anomali tespiti (API hazır olunca)
    - Zone bazlı karşılaştırmalar
    """)

# Footer
st.markdown("---")
st.markdown("""
**🏗️ Mimari:** FastAPI (Backend) + Streamlit (Frontend)  
**📁 Girdi:** İki Excel dosyası (Ana veri + Zone veri)  
**🎯 Çıktı:** Entegre risk analizi + AI insights
""")


