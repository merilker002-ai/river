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
# GITHUB-BASED MODEL MANAGER
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
            return None
        except Exception as e:
            st.sidebar.warning(f"⚠️ GitHub'dan model yüklenemedi: {e}")
            return None
    
    def upload_model(self, model: object, filepath: str = "models/river_model.pkl", 
                    commit_message: str = "Auto-update model") -> bool:
        """Modeli GitHub'a yükle"""
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
                st.sidebar.success("✅ Model GitHub'a yüklendi")
                return True
            else:
                st.sidebar.error(f"❌ Model yüklenemedi: {response.status_code}")
                return False
                
        except Exception as e:
            st.sidebar.error(f"❌ Model yükleme hatası: {e}")
            return False

# ======================================================================
# RIVER MODEL SERVICE (Lightweight - Bellek Optimize)
# ======================================================================
class RiverModelService:
    def __init__(self, github_manager: GitHubModelManager):
        self.github_manager = github_manager
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Modeli GitHub'dan yükle veya yeni oluştur"""
        self.model = self.github_manager.download_model()
        
        if self.model is None:
            # Yeni model oluştur
            try:
                from river import anomaly, preprocessing
                self.model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
                    n_estimators=25, 
                    height=8,
                    seed=42
                )
                st.sidebar.info("🆕 Yeni River modeli oluşturuldu")
            except ImportError:
                st.sidebar.warning("❌ River kütüphanesi kurulu değil")
                self.model = None
    
    def incremental_learn(self, data: List[Dict]) -> Dict:
        """Incremental learning yap"""
        if self.model is None:
            return {"status": "error", "message": "Model yok"}
        
        try:
            scores = []
            for record in data:
                # Feature extraction
                features = {
                    "tuketim": float(record.get('AKTIF_m3', 0)),
                    "gunluk_ort": float(record.get('GUNLUK_ORT_TUKETIM_m3', 0)),
                    "tutar": float(record.get('TOPLAM_TUTAR', 0))
                }
                
                # Score and learn
                score = self.model.score_one(features)
                self.model.learn_one(features)
                scores.append(score)
            
            # Modeli GitHub'a kaydet
            self.github_manager.upload_model(self.model)
            
            return {
                "status": "success",
                "processed_records": len(data),
                "avg_score": np.mean(scores) if scores else 0,
                "latest_scores": scores[-10:]  # Son 10 skor
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict(self, data: Dict) -> Dict:
        """Anomali skoru tahmini"""
        if self.model is None:
            return {"score": 0.0, "status": "error"}
        
        try:
            features = {
                "tuketim": float(data.get('AKTIF_m3', 0)),
                "gunluk_ort": float(data.get('GUNLUK_ORT_TUKETIM_m3', 0)),
                "tutar": float(data.get('TOPLAM_TUTAR', 0))
            }
            
            score = self.model.score_one(features)
            return {"score": score, "status": "success"}
        except:
            return {"score": 0.0, "status": "error"}

# ======================================================================
# VERİ İŞLEME FONKSİYONLARI
# ======================================================================
@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder"""
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

    # Davranış analizi
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Kısaltılmış davranış analizi fonksiyonu
    def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
        
        if len(tesisat_verisi) < 3:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta"

        tuketimler = tesisat_verisi['AKTIF_m3'].values
        
        # Basitleştirilmiş risk analizi
        sifir_sayisi = sum(tuketimler == 0)
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0
        
        risk_seviyesi = "Düşük"
        if sifir_sayisi >= 3 or varyasyon_katsayisi > 1.5 or tuketimler[-1] == 0:
            risk_seviyesi = "Yüksek"
        elif sifir_sayisi >= 1 or varyasyon_katsayisi > 0.8:
            risk_seviyesi = "Orta"

        yorumlar = ["Normal tüketim paterni"] if risk_seviyesi == "Düşük" else ["Tüketimde dalgalanma gözlemleniyor"]
        
        return np.random.choice(yorumlar), "Yok", risk_seviyesi

    # Tüm tesisatlar için davranış analizi
    davranis_sonuclari = []
    for idx, row in son_okumalar.iterrows():
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

        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi




# ======================================================================
# STREAMLIT ARAYÜZ - PROFESYONEL MİMARİ
# ======================================================================

# GitHub configuration - BUNLARI STREAMLIT CLOUD SECRETS'A EKLEYİN
GITHUB_OWNER = "your_username"  # GitHub kullanıcı adınız
GITHUB_REPO = "your_repo_name"  # Repo adı
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)  # Streamlit Cloud secrets

# Initialize services
github_manager = GitHubModelManager(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)
model_service = RiverModelService(github_manager)

st.set_page_config(
    page_title="Su Tüketim AI Analiz - GitHub + Streamlit",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Su Tüketim AI Analiz Sistemi")
st.markdown("🚀 **Profesyonel Mimari: GitHub + Streamlit + Incremental Learning**")

# Sidebar - Model Yönetimi
st.sidebar.header("🧠 AI Model Yönetimi")

# Model durumu
if model_service.model is not None:
    st.sidebar.success("✅ River Modeli Aktif")
else:
    st.sidebar.warning("⚠️ River Modeli Devre Dışı")

# Model işlemleri
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Modeli Güncelle"):
        model_service.load_model()
        st.rerun()

with col2:
    if st.button("🗑️ Modeli Sıfırla"):
        # GitHub'dan modeli sil (opsiyonel - manual yapılabilir)
        st.info("Modeli sıfırlamak için GitHub'dan models/river_model.pkl dosyasını silin")
        st.rerun()

# Dosya yükleme
st.sidebar.header("📁 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin", 
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Incremental Learning Kontrolü
st.sidebar.header("🔁 Incremental Learning")
auto_learn = st.sidebar.checkbox("Otomatik Öğrenme", value=True, 
                                help="Yeni veri yüklendiğinde otomatik öğren")

batch_size = st.sidebar.slider("Batch Boyutu", 10, 1000, 100, 
                              help="Aynı anda işlenecek kayıt sayısı")

# Demo verisi
if st.sidebar.button("🎮 Demo Modu"):
    # Demo verisi oluştur
    np.random.seed(42)
    demo_data = []
    for i in range(500):  # Daha küçük demo
        tesisat_no = f"TS{1000 + i}"
        aktif_m3 = np.random.gamma(2, 10)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0.1),
            'TOPLAM_TUTAR': aktif_m3 * 15,
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
        })
    
    df = pd.DataFrame(demo_data)
    son_okumalar = df.copy()
    son_okumalar['OKUMA_PERIYODU_GUN'] = 300
    son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
    
    risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.7, 0.2, 0.1])
    son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
    son_okumalar['DAVRANIS_YORUMU'] = "Demo verisi"
    son_okumalar['SUPHELI_DONEMLER'] = "Yok"
    
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum', 
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    
    st.success("✅ Demo verisi oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek veri yükleme
    df, son_okumalar, zone_analizi = load_and_analyze_data(uploaded_file, zone_file)
    
    # Incremental Learning
    if auto_learn and model_service.model is not None and df is not None:
        with st.sidebar:
            with st.spinner("🤖 AI öğreniyor..."):
                # Batch processing - belleği koru
                records = df.head(batch_size).to_dict('records')
                result = model_service.incremental_learn(records)
                
                if result["status"] == "success":
                    st.success(f"✅ {result['processed_records']} kayıt işlendi")
                    
                    # River skorlarını ekle
                    if 'RIVER_SCORE_MEAN' not in son_okumalar.columns:
                        # Tesisat bazında River skorları hesapla
                        river_scores = []
                        for _, row in son_okumalar.iterrows():
                            prediction = model_service.predict(row.to_dict())
                            river_scores.append(prediction['score'])
                        
                        son_okumalar['RIVER_SCORE'] = river_scores
                else:
                    st.error(f"❌ Öğrenme hatası: {result['message']}")
else:
    st.warning("⚠️ Lütfen Excel dosyasını yükleyin veya Demo modunu kullanın")
    st.stop()

# ======================================================================
# DASHBOARD GÖRSELLEŞTİRME
# ======================================================================

# Genel Metrikler
if son_okumalar is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Toplam Tesisat", f"{len(son_okumalar):,}")
    
    with col2:
        st.metric("💧 Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")
    
    with col3:
        st.metric("💰 Toplam Gelir", f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL")
    
    with col4:
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric("🚨 Yüksek Riskli", f"{yuksek_riskli}")

# Tab Menü
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Genel Görünüm", 
    "🗺️ Zone Analizi", 
    "🔍 Detaylı Analiz",
    "🤖 AI Insights"
])

with tab1:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                              title='Günlük Tüketim Dağılımı',
                              color_discrete_sequence=['#3498DB'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            color='RISK_SEVIYESI',
                            title='Tüketim-Tutar İlişkisi',
                            color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
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
                         title='Zone Bazlı Tesisat Sayısı')
            st.plotly_chart(fig4, use_container_width=True)

with tab3:
    if son_okumalar is not None:
        # Filtreleme ve detaylı analiz
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Filtreleme")
            risk_seviyeleri = st.multiselect(
                "Risk Seviyeleri",
                options=['Düşük', 'Orta', 'Yüksek'],
                default=['Yüksek', 'Orta']
            )
        
        with col2:
            filtreli_veri = son_okumalar[son_okumalar['RISK_SEVIYESI'].isin(risk_seviyeleri)]
            st.dataframe(
                filtreli_veri[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'RISK_SEVIYESI', 'DAVRANIS_YORUMU']].head(20),
                use_container_width=True
            )

with tab4:
    st.header("🤖 AI - River Model Insights")
    
    if son_okumalar is not None and 'RIVER_SCORE' in son_okumalar.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig5 = px.histogram(son_okumalar, x='RIVER_SCORE', 
                              title='River Anomali Skor Dağılımı',
                              nbins=30)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # En yüksek anomali skorlu tesisatlar
            high_anomaly = son_okumalar.nlargest(10, 'RIVER_SCORE')[['TESISAT_NO', 'RIVER_SCORE', 'AKTIF_m3', 'RISK_SEVIYESI']]
            st.dataframe(high_anomaly, use_container_width=True)
        
        # AI + Heuristic kombinasyonu
        st.subheader("🔥 Kombine Risk Analizi")
        son_okumalar['KOMBINE_RISK'] = np.where(
            (son_okumalar['RISK_SEVIYESI'] == 'Yüksek') | (son_okumalar['RIVER_SCORE'] > 0.7),
            'Yüksek', 
            np.where(
                (son_okumalar['RISK_SEVIYESI'] == 'Orta') | (son_okumalar['RIVER_SCORE'] > 0.4),
                'Orta', 
                'Düşük'
            )
        )
        
        fig6 = px.scatter(son_okumalar, x='AKTIF_m3', y='RIVER_SCORE',
                         color='KOMBINE_RISK', size='TOPLAM_TUTAR',
                         hover_data=['TESISAT_NO', 'DAVRANIS_YORUMU'],
                         title='AI + Heuristic Kombine Risk Analizi',
                         color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
        st.plotly_chart(fig6, use_container_width=True)
        
    else:
        st.info("🤖 AI analiz için veri yükleyin ve incremental learning'i aktif edin")

# Footer
st.markdown("---")
st.markdown("""
**🔧 Sistem Mimarisi:** 
- 🐍 Python + Streamlit 
- 🧠 River (Incremental ML) 
- 📁 GitHub Model Storage 
- ☁️ Streamlit Cloud Deploy
""")
