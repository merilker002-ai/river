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
import requests
import json
import base64
import io
from typing import Dict, List, Optional
import hashlib

warnings.filterwarnings('ignore')

# ======================================================================
# GITHUB MODEL MANAGER - Gerçek Incremental Learning için
# ======================================================================
class GitHubModelManager:
    def __init__(self, repo_owner: str, repo_name: str, token: str = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
        
    def _get_headers(self):
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers
    
    def download_model(self, filepath: str = "models/river_model.pkl") -> Optional[object]:
        """GitHub'dan modeli indir"""
        try:
            url = f"{self.base_url}/{filepath}"
            response = requests.get(url, headers=self._get_headers())
            
            if response.status_code == 200:
                content = response.json()
                if 'content' in content:
                    # Base64 decode
                    model_data = base64.b64decode(content['content'])
                    model = pickle.loads(model_data)
                    st.sidebar.success("✅ Model GitHub'dan yüklendi")
                    return model
            st.sidebar.info("ℹ️ GitHub'da model bulunamadı, yeni oluşturulacak")
            return None
        except Exception as e:
            st.sidebar.warning(f"⚠️ GitHub'dan model yüklenemedi: {e}")
            return None
    
    def upload_model(self, model: object, filepath: str = "models/river_model.pkl", 
                    commit_message: str = "Auto-update: Incremental learning") -> bool:
        """Modeli GitHub'a yükle - INCREMENTAL LEARNING İÇİN"""
        try:
            # Modeli serialize et
            model_bytes = pickle.dumps(model)
            model_b64 = base64.b64encode(model_bytes).decode()
            
            # Önce mevcut dosyayı kontrol et (SHA gerekli)
            url = f"{self.base_url}/{filepath}"
            response = requests.get(url, headers=self._get_headers())
            
            data = {
                "message": commit_message,
                "content": model_b64,
                "branch": "main"
            }
            
            if response.status_code == 200:
                existing_file = response.json()
                data["sha"] = existing_file["sha"]
            
            # Dosyayı yükle
            response = requests.put(url, headers=self._get_headers(), json=data)
            
            if response.status_code in [200, 201]:
                st.sidebar.success("✅ Model GitHub'a kaydedildi")
                return True
            else:
                st.sidebar.error(f"❌ Model yüklenemedi: {response.status_code}")
                return False
                
        except Exception as e:
            st.sidebar.error(f"❌ Model yükleme hatası: {e}")
            return False

# ======================================================================
# RIVER INCREMENTAL LEARNING SERVICE
# ======================================================================
class RiverIncrementalService:
    def __init__(self, github_manager: GitHubModelManager):
        self.github_manager = github_manager
        self.model = None
        self.learning_history = []
        self.load_model()
    
    def load_model(self):
        """Modeli GitHub'dan yükle veya yeni oluştur"""
        self.model = self.github_manager.download_model()
        
        if self.model is None:
            # Yeni model oluştur - INCREMENTAL LEARNING için optimize
            try:
                from river import anomaly, preprocessing
                # Hafıza dostu model - gerçek incremental learning
                self.model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
                    n_estimators=30,  # Daha az memory
                    height=10,        # Derinlik sınırlı
                    window_size=100,  # Streaming için
                    seed=42
                )
                st.sidebar.info("🆕 Yeni Incremental Learning modeli oluşturuldu")
                # Hemen GitHub'a kaydet
                self.github_manager.upload_model(self.model)
            except ImportError:
                st.sidebar.warning("❌ River kütüphanesi kurulu değil")
                self.model = None
    
    def incremental_learn_batch(self, data_batch: List[Dict]) -> Dict:
        """Batch incremental learning - BELLEK OPTİMİZE"""
        if self.model is None:
            return {"status": "error", "message": "Model yok"}
        
        try:
            scores = []
            processed_count = 0
            
            for record in data_batch:
                try:
                    # Feature extraction - gerçek zamanlı
                    features = {
                        "tuketim": float(record.get('AKTIF_m3', 0)),
                        "gunluk_ort": float(record.get('GUNLUK_ORT_TUKETIM_m3', 
                                                     record.get('AKTIF_m3', 0) / 30)),  # Fallback
                        "tutar": float(record.get('TOPLAM_TUTAR', 0))
                    }
                    
                    # INCREMENTAL LEARNING: Score + Learn
                    score = self.model.score_one(features)
                    self.model.learn_one(features)  # ✅ Bu satır incremental learning yapar
                    
                    scores.append(score)
                    processed_count += 1
                    
                except Exception as e:
                    continue
            
            # Modeli GitHub'a KAYDET - incremental state persist
            if processed_count > 0:
                success = self.github_manager.upload_model(self.model)
                if success:
                    # Learning history'yi güncelle
                    self.learning_history.append({
                        'timestamp': datetime.now(),
                        'processed': processed_count,
                        'avg_score': np.mean(scores) if scores else 0
                    })
            
            return {
                "status": "success",
                "processed_records": processed_count,
                "avg_score": np.mean(scores) if scores else 0,
                "model_updated": processed_count > 0
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_anomaly(self, data: Dict) -> float:
        """Anomali skoru tahmini"""
        if self.model is None:
            return 0.0
        
        try:
            features = {
                "tuketim": float(data.get('AKTIF_m3', 0)),
                "gunluk_ort": float(data.get('GUNLUK_ORT_TUKETIM_m3', 0)),
                "tutar": float(data.get('TOPLAM_TUTAR', 0))
            }
            return self.model.score_one(features)
        except:
            return 0.0
    
    def get_learning_stats(self):
        """Öğrenme istatistiklerini getir"""
        if not self.learning_history:
            return {"total_learned": 0, "last_update": "Never"}
        
        total = sum(h['processed'] for h in self.learning_history)
        last = self.learning_history[-1]['timestamp'].strftime("%Y-%m-%d %H:%M")
        return {"total_learned": total, "last_update": last}

# ======================================================================
# VERİ İŞLEME - Incremental Learning için optimize
# ======================================================================
@st.cache_data(ttl=3600)  # 1 saat cache
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder - INCREMENTAL READY"""
    try:
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
            st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

    # Davranış analizi - INCREMENTAL için optimize
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Hızlı davranış analizi - INCREMENTAL için basitleştirilmiş
    def quick_risk_analysis(tesisat_no, df):
        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
        
        if len(tesisat_verisi) < 2:
            return "Yetersiz veri", "Düşük"

        tuketimler = tesisat_verisi['AKTIF_m3'].values
        
        # Hızlı risk hesaplama
        sifir_sayisi = sum(tuketimler == 0)
        son_tuketim = tuketimler[-1] if len(tuketimler) > 0 else 0
        
        if sifir_sayisi >= 2 or son_tuketim == 0:
            return "Düzensiz tüketim", "Yüksek"
        elif sifir_sayisi >= 1:
            return "Ara sıra sıfır", "Orta"
        else:
            return "Normal patern", "Düşük"

    # Tüm tesisatlar için hızlı analiz
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

    # Zone analizi
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        # Son 3 ay verisi - INCREMENTAL için
        three_months_ago = datetime.now() - timedelta(days=90)
        recent_df = df[df['OKUMA_TARIHI'] >= three_months_ago]
        if len(recent_df) == 0:
            recent_df = df.copy()
        
        zone_analizi = recent_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi




#############################################################################################################################################

# ======================================================================
# STREAMLIT ARAYÜZ - INCREMENTAL LEARNING FOCUS
# ======================================================================

# GitHub configuration - STREAMLIT CLOUD SECRETS
GITHUB_OWNER = "merilker002-ai"  # YOUR GitHub username
GITHUB_REPO = "river"           # YOUR repo name
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)

# Initialize Incremental Learning Service
github_manager = GitHubModelManager(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)
river_service = RiverIncrementalService(github_manager)

st.set_page_config(
    page_title="🤖 Su Tüketim AI - Incremental Learning",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Su Tüketim AI Analiz Sistemi")
st.markdown("🚀 **Gerçek Incremental Learning + GitHub Persistence**")

# ======================================================================
# SIDEBAR - INCREMENTAL LEARNING KONTROL
# ======================================================================
st.sidebar.header("🧠 Incremental Learning AI")

# Model durumu
model_stats = river_service.get_learning_stats()
st.sidebar.metric("📚 Toplam Öğrenilen", f"{model_stats['total_learned']} kayıt")
st.sidebar.metric("🕒 Son Güncelleme", model_stats['last_update'])

if river_service.model is not None:
    st.sidebar.success("✅ AI Model Aktif - Incremental Learning Hazır")
else:
    st.sidebar.error("❌ AI Model Devre Dışı")

# Incremental Learning Kontrolleri
st.sidebar.header("🔧 Learning Kontrol")
auto_learn = st.sidebar.checkbox("🔄 OTOMATİK Incremental Learning", value=True,
                                help="Yeni veri geldikçe otomatik öğren")

learning_mode = st.sidebar.selectbox(
    "🎯 Learning Modu",
    ["Yüksek Performans", "Düşük Bellek", "Maximum Accuracy"],
    help="Öğrenme modunu seçin"
)

batch_size = st.sidebar.slider("📦 Batch Boyutu", 50, 500, 200,
                              help="Aynı anda işlenecek kayıt sayısı")

# Manuel learning kontrolü
if st.sidebar.button("🎓 Manuel Öğrenme Başlat"):
    if river_service.model is not None:
        st.sidebar.info("Manuel öğrenme butonu - veri yükleyin")

# Model yönetimi
if st.sidebar.button("🔄 Modeli Yeniden Yükle"):
    river_service.load_model()
    st.rerun()

# ======================================================================
# DOSYA YÜKLEME - INCREMENTAL LEARNING TRIGGER
# ======================================================================
st.sidebar.header("📁 Veri Yükleme")

uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin",
    type=["xlsx"],
    help="Yeni veri yükleyin - incremental learning otomatik başlar"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin", 
    type=["xlsx"],
    help="Zone bilgileri (opsiyonel)"
)

# ======================================================================
# DEMO VERİSİ - INCREMENTAL LEARNING TEST
# ======================================================================
if st.sidebar.button("🧪 Demo + Incremental Learning Test"):
    with st.spinner("Demo verisi oluşturuluyor ve AI öğreniyor..."):
        # Akıllı demo verisi - incremental learning test
        np.random.seed(42)
        demo_data = []
        
        # Gerçekçi tüketim patternleri
        patterns = {
            'normal': lambda: max(np.random.normal(15, 5), 1),
            'high_var': lambda: max(np.random.normal(20, 15), 0),
            'zero_pattern': lambda: 0 if np.random.random() < 0.3 else max(np.random.normal(10, 3), 1)
        }
        
        for i in range(300):  # Optimize boyut
            pattern_type = np.random.choice(['normal', 'high_var', 'zero_pattern'], p=[0.7, 0.2, 0.1])
            tuketim = patterns[pattern_type]()
            
            demo_data.append({
                'TESISAT_NO': f"DEMO_{1000 + i}",
                'AKTIF_m3': tuketim,
                'TOPLAM_TUTAR': tuketim * 12 + np.random.normal(0, 5),
                'ILK_OKUMA_TARIHI': pd.Timestamp('2024-01-01'),
                'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
                'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
            })
        
        df = pd.DataFrame(demo_data)
        
        # Davranış analizi
        son_okumalar = df.copy()
        son_okumalar['OKUMA_PERIYODU_GUN'] = 300
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        
        # Risk analizi
        risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.6, 0.3, 0.1])
        son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
        son_okumalar['DAVRANIS_YORUMU'] = "Demo analiz"
        
        # Zone analizi
        zone_analizi = df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum', 
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        
        # ✅ INCREMENTAL LEARNING TEST - Demo verisiyle
        if river_service.model is not None and auto_learn:
            demo_batch = df.head(batch_size).to_dict('records')
            learn_result = river_service.incremental_learn_batch(demo_batch)
            
            if learn_result["status"] == "success":
                st.sidebar.success(f"🧠 Demo ile öğrenildi: {learn_result['processed_records']} kayıt")
                
                # River skorlarını ekle
                river_scores = []
                for _, row in son_okumalar.iterrows():
                    score = river_service.predict_anomaly(row.to_dict())
                    river_scores.append(score)
                
                son_okumalar['RIVER_SCORE'] = river_scores
            else:
                st.sidebar.error(f"❌ Demo learning hatası: {learn_result['message']}")
        
        st.success("✅ Demo verisi oluşturuldu ve AI öğrendi!")

# ======================================================================
# GERÇEK VERİ İŞLEME - INCREMENTAL LEARNING
# ======================================================================
elif uploaded_file is not None:
    # Gerçek veri yükleme
    df, son_okumalar, zone_analizi = load_and_analyze_data(uploaded_file, zone_file)
    
    # ✅ GERÇEK INCREMENTAL LEARNING - Yeni veri geldiğinde
    if auto_learn and river_service.model is not None and df is not None:
        with st.sidebar:
            with st.spinner("🤖 AI yeni veriyi öğreniyor..."):
                # Bellek optimizasyonu - batch processing
                records_to_learn = df.head(batch_size).to_dict('records')
                learn_result = river_service.incremental_learn_batch(records_to_learn)
                
                if learn_result["status"] == "success":
                    if learn_result["processed_records"] > 0:
                        st.success(f"✅ {learn_result['processed_records']} kayıt öğrenildi")
                        
                        # Tüm veriye River skorlarını uygula
                        river_scores = []
                        for _, row in son_okumalar.iterrows():
                            score = river_service.predict_anomaly(row.to_dict())
                            river_scores.append(score)
                        
                        son_okumalar['RIVER_SCORE'] = river_scores
                        son_okumalar['RIVER_RISK'] = np.where(
                            son_okumalar['RIVER_SCORE'] > 0.6, 'Yüksek',
                            np.where(son_okumalar['RIVER_SCORE'] > 0.3, 'Orta', 'Düşük')
                        )
                    else:
                        st.info("ℹ️ Öğrenilecek yeni kayıt yok")
                else:
                    st.error(f"❌ Öğrenme hatası: {learn_result['message']}")

else:
    st.warning("⚠️ Lütfen Excel dosyasını yükleyin veya Demo modunu kullanın")
    st.stop()

# ======================================================================
# DASHBOARD - INCREMENTAL LEARNING RESULTS
# ======================================================================

# Real-time Metrics
st.header("📊 Gerçek Zamanlı Metrikler - Incremental Learning")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏠 Toplam Tesisat", f"{len(son_okumalar):,}")

with col2:
    st.metric("💧 Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")

with col3:
    st.metric("🎯 Geleneksel Risk", 
             f"{(son_okumalar['RISK_SEVIYESI'] == 'Yüksek').sum()} tesisat")

with col4:
    river_high_risk = len(son_okumalar[son_okumalar.get('RIVER_RISK', 'Düşük') == 'Yüksek'])
    st.metric("🤖 AI Risk", f"{river_high_risk} tesisat")

# INCREMENTAL LEARNING INSIGHTS
st.header("🧠 Incremental Learning Insights")

if 'RIVER_SCORE' in son_okumalar.columns:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 AI Analiz", "🎯 Risk Karşılaştırma", "📊 Learning Stats", "🚨 Anomaliler"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(son_okumalar, x='RIVER_SCORE', 
                              title='AI Anomali Skor Dağılımı',
                              nbins=30, color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # En yüksek anomali skorlular
            high_anomaly = son_okumalar.nlargest(10, 'RIVER_SCORE')[
                ['TESISAT_NO', 'RIVER_SCORE', 'AKTIF_m3', 'RISK_SEVIYESI', 'RIVER_RISK']
            ]
            st.dataframe(high_anomaly, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Geleneksel vs AI risk karşılaştırması
            traditional_risk = son_okumalar['RISK_SEVIYESI'].value_counts()
            fig2 = px.pie(values=traditional_risk.values, names=traditional_risk.index,
                         title='🎯 Geleneksel Risk Dağılımı',
                         color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            if 'RIVER_RISK' in son_okumalar.columns:
                ai_risk = son_okumalar['RIVER_RISK'].value_counts()
                fig3 = px.pie(values=ai_risk.values, names=ai_risk.index,
                             title='🤖 AI Risk Dağılımı',
                             color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
                st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("📚 Learning İstatistikleri")
        stats = river_service.get_learning_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📖 Toplam Öğrenilen", stats['total_learned'])
        with col2:
            st.metric("🕒 Son Öğrenme", stats['last_update'])
        with col3:
            st.metric("🎯 Model Durumu", "Aktif" if river_service.model else "Pasif")
        
        # Learning history grafiği
        if river_service.learning_history:
            history_df = pd.DataFrame(river_service.learning_history)
            fig4 = px.line(history_df, x='timestamp', y='processed',
                          title='📈 Incremental Learning Geçmişi',
                          labels={'timestamp': 'Zaman', 'processed': 'İşlenen Kayıt'})
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab4:
        st.subheader("🚨 AI Tespit Edilen Anomaliler")
        
        # Kombine risk analizi
        high_risk_combined = son_okumalar[
            (son_okumalar['RISK_SEVIYESI'] == 'Yüksek') | 
            (son_okumalar.get('RIVER_RISK', 'Düşük') == 'Yüksek')
        ]
        
        if len(high_risk_combined) > 0:
            st.success(f"🎯 {len(high_risk_combined)} adet yüksek riskli tesisat tespit edildi")
            
            fig5 = px.scatter(high_risk_combined, x='AKTIF_m3', y='RIVER_SCORE',
                             color='RISK_SEVIYESI', size='TOPLAM_TUTAR',
                             hover_data=['TESISAT_NO', 'DAVRANIS_YORUMU'],
                             title='🔥 Yüksek Riskli Tesisatlar - AI + Geleneksel',
                             color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
            st.plotly_chart(fig5, use_container_width=True)
            
            # Detaylı liste
            st.dataframe(high_risk_combined[
                ['TESISAT_NO', 'AKTIF_m3', 'RIVER_SCORE', 'RISK_SEVIYESI', 'RIVER_RISK', 'DAVRANIS_YORUMU']
            ].sort_values('RIVER_SCORE', ascending=False), use_container_width=True)
        else:
            st.info("🎉 Hiç yüksek riskli tesisat bulunamadı!")

else:
    st.info("🤖 AI analiz için veri yükleyin - incremental learning otomatik başlayacak")

# Footer
st.markdown("---")
st.markdown("""
**🔧 Profesyonel Mimari:** 
- 🐍 Streamlit Cloud 
- 🧠 River Incremental Learning 
- 📁 GitHub Model Storage 
- 🔄 Otomatik Model Persistence
- 💾 Bellek Optimizasyonu
""")
