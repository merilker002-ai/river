import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re
from model_api import analiz_modeli

warnings.filterwarnings('ignore')

# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Su Tüketim Davranış Analiz Dashboard",
    page_icon="💧",
    layout="wide"
)

# ======================================================================
# 📊 GELİŞMİŞ VERİ İŞLEME FONKSİYONLARI
# ======================================================================

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve gelişmiş analiz eder"""
    try:
        # Ana veri dosyasını oku
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None, None

    # Model ile veri ön işleme
    df = analiz_modeli.veri_on_isleme(df)
    
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

    # Son okumaları al
    son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()

    # Tüm tesisatlar için GELİŞMİŞ davranış analizi yap
    st.info("🔍 Gelişmiş davranış analizi yapılıyor...")
    progress_bar = st.progress(0)
    davranis_sonuclari = []
    
    total_tesisat = len(son_okumalar)
    for i, (idx, row) in enumerate(son_okumalar.iterrows()):
        tesisat_verisi = df[df['TESISAT_NO'] == row['TESISAT_NO']].sort_values('OKUMA_TARIHI')
        yorum, supheli_donemler, risk, risk_puan = analiz_modeli.gelismis_davranis_analizi(tesisat_verisi)
        
        davranis_sonuclari.append({
            'TESISAT_NO': row['TESISAT_NO'],
            'DAVRANIS_YORUMU': yorum,
            'SUPHELI_DONEMLER': supheli_donemler,
            'RISK_SEVIYESI': risk,
            'RISK_PUANI': risk_puan
        })
        
        # Progress bar güncelleme
        if i % 100 == 0:
            progress_bar.progress(min((i + 1) / total_tesisat, 1.0))

    progress_bar.progress(1.0)
    davranis_df = pd.DataFrame(davranis_sonuclari)
    son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    # Anomali tespiti
    st.info("🎯 Anomali tespiti yapılıyor...")
    son_okumalar = analiz_modeli.anomaly_detection(son_okumalar)

    # GELİŞMİŞ Zone analizi (son 3 ay)
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        # Son 3 aylık veriyi al
        son_tarih = df['OKUMA_TARIHI'].max()
        uc_ay_once = son_tarih - timedelta(days=90)
        son_uc_ay_df = df[df['OKUMA_TARIHI'] >= uc_ay_once]
        
        if len(son_uc_ay_df) == 0:
            son_uc_ay_df = df.copy()
        
        zone_analizi = son_uc_ay_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        # Gelişmiş Zone risk analizi
        son_uc_ay_risk = son_uc_ay_df.merge(son_okumalar[['TESISAT_NO', 'RISK_SEVIYESI']], on='TESISAT_NO', how='left')
        
        zone_risk_analizi = son_uc_ay_risk.groupby('KARNE_NO').agg({
            'RISK_SEVIYESI': lambda x: (x == 'Yüksek').sum(),
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
    """Gelişmiş demo verisi oluşturur"""
    np.random.seed(42)
    
    # Gelişmiş örnek veri oluştur
    demo_data = []
    tesisat_sayisi = 1500
    
    for i in range(tesisat_sayisi):
        tesisat_no = f"TS{1000 + i}"
        
        # Farklı tüketim patternleri oluştur
        pattern_type = np.random.choice(['normal', 'sifir_aralikli', 'yuksek_dalgalanma', 'artis_trend'], 
                                      p=[0.6, 0.15, 0.15, 0.1])
        
        if pattern_type == 'normal':
            aktif_m3 = np.random.gamma(2, 8)
        elif pattern_type == 'sifir_aralikli':
            aktif_m3 = 0 if np.random.random() < 0.3 else np.random.gamma(2, 6)
        elif pattern_type == 'yuksek_dalgalanma':
            aktif_m3 = np.random.gamma(5, 15)
        else:  # artis_trend
            aktif_m3 = np.random.gamma(4, 12)
        
        toplam_tutar = aktif_m3 * 15 + np.random.normal(0, 8)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Gelişmiş davranış analizi için çoklu okuma verisi oluştur
    coklu_okuma_data = []
    for tesisat in demo_data:
        tesisat_no = tesisat['TESISAT_NO']
        base_consumption = tesisat['AKTIF_m3']
        
        # 6 aylık veri oluştur
        for month in range(6):
            month_date = pd.Timestamp(f'2024-{5+month:02d}-15')
            noise = np.random.normal(0, base_consumption * 0.3)
            
            if 'sifir_aralikli' in tesisat_no and month % 3 == 0:
                consumption = 0
            elif 'yuksek_dalgalanma' in tesisat_no:
                consumption = max(base_consumption + np.random.normal(0, base_consumption * 0.8), 0)
            elif 'artis_trend' in tesisat_no:
                consumption = base_consumption * (1 + month * 0.15)
            else:
                consumption = max(base_consumption + noise, 0)
            
            coklu_okuma_data.append({
                'TESISAT_NO': tesisat_no,
                'AKTIF_m3': consumption,
                'TOPLAM_TUTAR': consumption * 15 + np.random.normal(0, 5),
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
st.markdown("**Gelişmiş Analiz Sistemi | 🔥 Ateş Böceği Görünümü**")

# Dosya yükleme bölümü
st.sidebar.header("📁 İki Dosya Yükle")
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

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    st.info("Demo modu aktif! Gelişmiş analiz ile çalışılıyor...")
    
    df_detayli = create_demo_data()
    df_detayli = analiz_modeli.veri_on_isleme(df_detayli)
    
    # Son okumaları al
    son_okumalar = df_detayli.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
    
    # Demo için analiz yap
    st.info("🔍 Demo verisi için gelişmiş davranış analizi yapılıyor...")
    
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
        'ZONE1': {'ad': 'ÖLÇÜM NOKTASI-1 (KIRMIZI)', 'verilen_su': 20078.00, 'tahakkuk_m3': 7010.00, 'kayip_oran': 65.09},
        'ZONE2': {'ad': 'ÖLÇÜM NOKTASI-2 (MAVİ)', 'verilen_su': 3968.00, 'tahakkuk_m3': 1813.00, 'kayip_oran': 54.31},
        'ZONE3': {'ad': 'ÖLÇÜM NOKTASI-3 (ALT BÖLGE) (YEŞİL)', 'verilen_su': 19623.00, 'tahakkuk_m3': 7375.00, 'kayip_oran': 62.42},
        'ZONE4': {'ad': 'ÖLÇÜM NOKTASI-5 (ÜST BÖLGE) (MOR)', 'verilen_su': 18666.00, 'tahakkuk_m3': 7654.00, 'kayip_oran': 58.99},
        'ZONE5': {'ad': 'HASTANE BÖLGESİ (SARI)', 'verilen_su': 17775.00, 'tahakkuk_m3': 2134.00, 'kayip_oran': 87.99}
    }
    
    st.success("✅ Gelişmiş demo verisi başarıyla oluşturuldu!")
    df = df_detayli

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Genel Metrikler
if son_okumalar is not None:
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric(
            label="🚨 Yüksek Riskli",
            value=f"{yuksek_riskli}"
        )
    
    with col5:
        anomali_sayisi = len(son_okumalar[son_okumalar['ANOMALY_SCORE'] == -1])
        st.metric(
            label="🎯 Anomali Tespit",
            value=f"{anomali_sayisi}"
        )

# Tab Menü
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
            # Tüketim Dağılım Grafiği
            fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                              title='Günlük Tüketim Dağılımı',
                              labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'},
                              color_discrete_sequence=['#3498DB'])
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Tüketim-Tutar İlişkisi (Risk Renkli)
            fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            color='RISK_SEVIYESI',
                            title='Tüketim-Tutar İlişkisi (Risk Seviyeli)',
                            labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                            color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig2, use_container_width=True)
        
        # Zaman Serisi Grafiği
        if df is not None:
            df_aylik = df.groupby(df['OKUMA_TARIHI'].dt.to_period('M')).agg({
                'AKTIF_m3': 'sum',
                'TOPLAM_TUTAR': 'sum'
            }).reset_index()
            df_aylik['OKUMA_TARIHI'] = df_aylik['OKUMA_TARIHI'].dt.to_timestamp()

            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(
                go.Scatter(x=df_aylik['OKUMA_TARIHI'], y=df_aylik['AKTIF_m3'], 
                          name="Tüketim (m³)", line=dict(color='blue')),
                secondary_y=False,
            )
            fig3.add_trace(
                go.Scatter(x=df_aylik['OKUMA_TARIHI'], y=df_aylik['TOPLAM_TUTAR'], 
                          name="Gelir (TL)", line=dict(color='green')),
                secondary_y=True,
            )
            fig3.update_layout(title_text="Aylık Tüketim ve Gelir Trendi")
            fig3.update_xaxes(title_text="Tarih")
            fig3.update_yaxes(title_text="Tüketim (m³)", secondary_y=False)
            fig3.update_yaxes(title_text="Gelir (TL)", secondary_y=True)
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    if zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone Tüketim Dağılımı
            fig4 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                        title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Zone Risk Dağılımı
            fig5 = px.bar(zone_analizi, x='KARNE_NO', y='YUKSEK_RISK_ORANI',
                        title='Zone Bazlı Yüksek Risk Oranı (%)',
                        labels={'KARNE_NO': 'Zone', 'YUKSEK_RISK_ORANI': 'Yüksek Risk Oranı %'},
                        color='YUKSEK_RISK_ORANI',
                        color_continuous_scale='reds')
            st.plotly_chart(fig5, use_container_width=True)
        
        # Zone Karşılaştırma Tablosu
        st.subheader("Zone Karşılaştırma Tablosu")
        zone_karsilastirma = zone_analizi[['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR', 'YUKSEK_RISK_ORANI']].copy()
        if 'ad' in zone_analizi.columns:
            zone_karsilastirma['Zone Adı'] = zone_analizi['ad']
        if 'verilen_su' in zone_analizi.columns:
            zone_karsilastirma['Verilen Su (m³)'] = zone_analizi['verilen_su']
            zone_karsilastirma['Tahakkuk (m³)'] = zone_analizi['tahakkuk_m3']
            zone_karsilastirma['Kayıp Oranı (%)'] = zone_analizi['kayip_oran']
        
        st.dataframe(zone_karsilastirma.style.format({
            'TOPLAM_TUKETIM': '{:,.0f}',
            'TOPLAM_GELIR': '{:,.0f}',
            'YUKSEK_RISK_ORANI': '{:.1f}%',
            'Kayıp Oranı (%)': '{:.1f}%'
        }), use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    if son_okumalar is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Filtreleme Seçenekleri")
            
            # Tüketim Slider
            max_tuketim = int(son_okumalar['AKTIF_m3'].max()) if len(son_okumalar) > 0 else 100
            tuketim_range = st.slider(
                "Tüketim Aralığı (m³)",
                min_value=0,
                max_value=max_tuketim,
                value=[0, min(100, max_tuketim)],
                help="Tüketim değerine göre filtreleme yapın"
            )
            
            # Risk Seviyesi Filtresi
            risk_seviyeleri = st.multiselect(
                "Risk Seviyeleri",
                options=['Düşük', 'Orta', 'Yüksek'],
                default=['Yüksek', 'Orta']
            )
            
            # Anomali Filtresi
            anomali_filtre = st.checkbox("Sadece Anomalileri Göster", value=False)
            
            # Sıralama Seçeneği
            siralama = st.selectbox(
                "Sıralama Türü",
                options=['En Yüksek Risk', 'En Yüksek Tüketim', 'En Düşük Tüketim'],
                index=0
            )
        
        with col2:
            st.subheader("Tesisat Tablosu")
            
            # Filtreleme
            min_tuketim, max_tuketim = tuketim_range
            filtreli_veri = son_okumalar[
                (son_okumalar['AKTIF_m3'] >= min_tuketim) & 
                (son_okumalar['AKTIF_m3'] <= max_tuketim) &
                (son_okumalar['RISK_SEVIYESI'].isin(risk_seviyeleri))
            ]
            
            # Anomali filtresi
            if anomali_filtre:
                filtreli_veri = filtreli_veri[filtreli_veri['ANOMALY_SCORE'] == -1]
            
            # Sıralama
            if siralama == 'En Yüksek Tüketim':
                gosterilecek_veri = filtreli_veri.nlargest(20, 'AKTIF_m3')
            elif siralama == 'En Düşük Tüketim':
                gosterilecek_veri = filtreli_veri.nsmallest(20, 'AKTIF_m3')
            else:
                # Risk önceliğine göre sırala
                risk_sirasi = {'Yüksek': 3, 'Orta': 2, 'Düşük': 1}
                filtreli_veri['RISK_SIRASI'] = filtreli_veri['RISK_SEVIYESI'].map(risk_sirasi)
                gosterilecek_veri = filtreli_veri.nlargest(20, ['RISK_SIRASI', 'RISK_PUANI', 'AKTIF_m3'])
            
            # Tablo gösterimi
            st.dataframe(
                gosterilecek_veri[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'GUNLUK_ORT_TUKETIM_m3', 'RISK_SEVIYESI', 'RISK_PUANI', 'DAVRANIS_YORUMU']].round(3),
                use_container_width=True
            )

with tab4:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Dağılımı
            risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
            fig6 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                         title='Risk Seviyeleri Dağılımı',
                         color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            # Anomali Dağılımı
            anomali_dagilim = son_okumalar['ANOMALY_SCORE'].value_counts()
            fig7 = px.pie(values=anomali_dagilim.values, names=['Normal', 'Anomali'],
                         title='Anomali Dağılımı',
                         color_discrete_sequence=['green', 'red'])
            st.plotly_chart(fig7, use_container_width=True)
        
        # Korelasyon Matrisi
        numeric_cols = son_okumalar.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr_matrix = son_okumalar[numeric_cols].corr()
            
            fig8 = px.imshow(corr_matrix, 
                           title='Korelasyon Matrisi',
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
            st.plotly_chart(fig8, use_container_width=True)
        
        # Aykırı Değer Analizi
        fig9 = px.box(son_okumalar, y='AKTIF_m3', 
                     title='Tüketim Dağılımı - Aykırı Değer Analizi',
                     color_discrete_sequence=['#F39C12'])
        st.plotly_chart(fig9, use_container_width=True)

with tab5:
    st.header("🔥 Ateş Böceği Görünümü - Şüpheli Tesisatlar")
    
    if son_okumalar is not None:
        # Yüksek riskli tesisatları filtrele
        yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
        
        if len(yuksek_riskli) > 0:
            st.success(f"🚨 {len(yuksek_riskli)} adet yüksek riskli tesisat tespit edildi!")
            
            # Ateş böceği efekti için özel scatter plot
            fig10 = px.scatter(yuksek_riskli, x='AKTIF_m3', y='TOPLAM_TUTAR',
                             size='RISK_PUANI',
                             color='RISK_PUANI',
                             hover_name='TESISAT_NO',
                             hover_data=['DAVRANIS_YORUMU', 'SUPHELI_DONEMLER'],
                             title='🔥 Ateş Böceği Görünümü - Yüksek Riskli Tesisatlar',
                             labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                             color_continuous_scale='reds',
                             size_max=30)
            
            # Ateş böceği efekti için animasyon
            fig10.update_traces(marker=dict(symbol='star', line=dict(width=2, color='DarkOrange')),
                              selector=dict(mode='markers'))
            
            st.plotly_chart(fig10, use_container_width=True)
            
            # Detaylı liste
            st.subheader("Yüksek Riskli Tesisat Detayları")
            for idx, row in yuksek_riskli.iterrows():
                with st.expander(f"🚨 Tesisat No: {row['TESISAT_NO']} - Risk Puanı: {row['RISK_PUANI']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tüketim", f"{row['AKTIF_m3']:.1f} m³")
                    with col2:
                        st.metric("Tutar", f"{row['TOPLAM_TUTAR']:.1f} TL")
                    with col3:
                        st.metric("Günlük Ort.", f"{row['GUNLUK_ORT_TUKETIM_m3']:.3f} m³")
                    with col4:
                        st.metric("Anomali", "✅" if row['ANOMALY_SCORE'] == -1 else "❌")
                    
                    st.write(f"**Şüpheli Dönemler:** {row['SUPHELI_DONEMLER']}")
                    st.write(f"**Davranış Yorumu:** {row['DAVRANIS_YORUMU']}")
        else:
            st.info("🎉 Hiç yüksek riskli tesisat bulunamadı!")
        
        # Anomali tespit edilen tesisatlar
        anomaliler = son_okumalar[son_okumalar['ANOMALY_SCORE'] == -1]
        if len(anomaliler) > 0:
            st.subheader(f"🎯 Anomali Tespit Edilen Tesisatlar ({len(anomaliler)} adet)")
            
            fig11 = px.scatter(anomaliler, x='AKTIF_m3', y='TOPLAM_TUTAR',
                             color='RISK_SEVIYESI',
                             hover_name='TESISAT_NO',
                             title='Anomali Tespit Edilen Tesisatlar',
                             color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            
            st.plotly_chart(fig11, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💧 **Gelişmiş Su Tüketim Analiz Sistemi** | Streamlit Dashboard | 🔥 **Ateş Böceği Görünümü**")
