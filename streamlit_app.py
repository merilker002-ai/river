# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import time

# ======================================================================
# API CLIENT - PROFESYONEL
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
        """API sağlık kontrolü"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def incremental_learn(self, data: list, batch_id: str = None) -> dict:
        """Incremental learning isteği"""
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
        """Tahmin isteği"""
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
        """Model bilgileri"""
        try:
            response = self.session.get(f"{self.base_url}/model-info", timeout=5)
            return response.json()
        except:
            return {"status": "error"}

# ======================================================================
# STREAMLIT APP - TEMİZ VE PROFESYONEL
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
        st.sidebar.metric("💾 Bellek", model_info.get('stats', {}).get('memory_usage', '0 KB'))
else:
    st.sidebar.error("❌ API Bağlantısı Yok")
    st.sidebar.info("🔧 FastAPI servisini başlatın: `python model_api.py`")

# ======================================================================
# VERİ YÜKLEME VE INCREMENTAL LEARNING
# ======================================================================
st.sidebar.header("📁 Veri İşleme")

uploaded_file = st.sidebar.file_uploader(
    "Excel dosyası yükle",
    type=["xlsx"],
    help="Yeni veri yükleyin - incremental learning otomatik başlar"
)

# Learning ayarları
st.sidebar.header("🎯 Learning Kontrol")
auto_learn = st.sidebar.checkbox("🔄 Otomatik Incremental Learning", value=True)
batch_size = st.sidebar.slider("📦 Batch Boyutu", 100, 2000, 500)

# ======================================================================
# VERİ İŞLEME FONKSİYONU
# ======================================================================
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Veriyi yükle ve temizle"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Temel temizlik
        df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
        df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
        df = df[df['TESISAT_NO'].notnull()]
        
        # Günlük tüketim hesapla
        df['OKUMA_PERIYODU_GUN'] = (df['OKUMA_TARIHI'] - df['ILK_OKUMA_TARIHI']).dt.days
        df['OKUMA_PERIYODU_GUN'] = df['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        df['GUNLUK_ORT_TUKETIM_m3'] = df['AKTIF_m3'] / df['OKUMA_PERIYODU_GUN']
        df['GUNLUK_ORT_TUKETIM_m3'] = df['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

# ======================================================================
# ANA UYGULAMA LOGIC
# ======================================================================
if uploaded_file is not None:
    # Veriyi yükle
    with st.spinner("📊 Veri yükleniyor..."):
        df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ {len(df)} kayıt yüklendi")
        
        # INCREMENTAL LEARNING - API üzerinden
        if auto_learn and api_client.health_check():
            with st.spinner("🤖 AI yeni veriyi öğreniyor..."):
                # Batch processing - memory efficient
                batch_data = df.head(batch_size).to_dict('records')
                learn_result = api_client.incremental_learn(batch_data)
                
                if learn_result.get("status") == "success":
                    st.success(f"🎯 {learn_result['processed']} kayıt öğrenildi | Bellek: {learn_result['memory_usage']}")
                else:
                    st.error(f"❌ Öğrenme hatası: {learn_result.get('message', 'Bilinmeyen hata')}")
        
        # ======================================================================
        # ANALIZ VE GÖRSELLEŞTİRME
        # ======================================================================
        
        # Son okumaları al
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        
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
            # Geleneksel risk (basit heuristic)
            son_okumalar['GELENEKSEL_RISK'] = np.where(
                son_okumalar['AKTIF_m3'] == 0, 'Yüksek',
                np.where(son_okumalar['GUNLUK_ORT_TUKETIM_m3'] > 10, 'Orta', 'Düşük')
            )
            geleneksel_yuksek = (son_okumalar['GELENEKSEL_RISK'] == 'Yüksek').sum()
            st.metric("🎯 Geleneksel Risk", geleneksel_yuksek)
        
        with col4:
            if 'AI_RISK' in son_okumalar.columns:
                ai_yuksek = (son_okumalar['AI_RISK'] == 'Yüksek').sum()
                st.metric("🤖 AI Risk", ai_yuksek)
            else:
                st.metric("🤖 AI", "Pasif")
        
        # GÖRSELLEŞTİRMELER
        tab1, tab2, tab3 = st.tabs(["📈 Temel Analiz", "🤖 AI Insights", "🚨 Risk Karşılaştırma"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                                  title='Günlük Tüketim Dağılımı')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                                color='GELENEKSEL_RISK',
                                title='Tüketim-Tutar İlişkisi')
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            if 'AI_SKOR' in son_okumalar.columns:
                col1, col2 = st.columns(2)
                with col1:
                    fig3 = px.histogram(son_okumalar, x='AI_SKOR', 
                                      title='AI Anomali Skor Dağılımı',
                                      nbins=30)
                    st.plotly_chart(fig3, use_container_width=True)
                with col2:
                    # En yüksek AI riskliler
                    high_ai_risk = son_okumalar[son_okumalar['AI_RISK'] == 'Yüksek']
                    if len(high_ai_risk) > 0:
                        st.dataframe(high_ai_risk[
                            ['TESISAT_NO', 'AI_SKOR', 'AKTIF_m3', 'GELENEKSEL_RISK']
                        ].head(10), use_container_width=True)
                    else:
                        st.success("🎉 AI yüksek risk bulamadı!")
            else:
                st.info("🤖 AI analiz için API bağlantısı gerekli")
        
        with tab3:
            if 'AI_RISK' in son_okumalar.columns:
                col1, col2 = st.columns(2)
                with col1:
                    geleneksel_dagilim = son_okumalar['GELENEKSEL_RISK'].value_counts()
                    fig4 = px.pie(values=geleneksel_dagilim.values, 
                                names=geleneksel_dagilim.index,
                                title='Geleneksel Risk Dağılımı')
                    st.plotly_chart(fig4, use_container_width=True)
                with col2:
                    ai_dagilim = son_okumalar['AI_RISK'].value_counts()
                    fig5 = px.pie(values=ai_dagilim.values, 
                                names=ai_dagilim.index,
                                title='AI Risk Dağılımı')
                    st.plotly_chart(fig5, use_container_width=True)
                
                # Uyumsuzluk analizi
                uyumsuz = son_okumalar[
                    (son_okumalar['GELENEKSEL_RISK'] == 'Düşük') & 
                    (son_okumalar['AI_RISK'] == 'Yüksek')
                ]
                if len(uyumsuz) > 0:
                    st.warning(f"🚨 AI'nın tespit ettiği {len(uyumsuz)} gizli risk!")
        
        # DETAYLI LİSTE
        st.subheader("📋 Detaylı Tesisat Listesi")
        st.dataframe(son_okumalar[
            ['TESISAT_NO', 'AKTIF_m3', 'GUNLUK_ORT_TUKETIM_m3', 'GELENEKSEL_RISK', 'AI_RISK', 'AI_SKOR']
        ].sort_values('AI_SKOR', ascending=False).head(20), use_container_width=True)

else:
    # LANDING PAGE
    st.info("👆 Lütfen Excel dosyası yükleyin")
    
    # Demo butonu
    if st.button("🧪 Demo Modu"):
        st.info("Demo modu - gerçek veri yükleyin")

# Footer
st.markdown("---")
st.markdown("""
**🏗️ Mimari:** FastAPI (Backend) + Streamlit (Frontend) + River (Incremental AI)
**🔗 GitHub:** Model persistence otomatik
**💾 Bellek:** Optimize batch processing
**🚀 Ölçeklenebilir:** Microservice mimarisi
""")