import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class SuTuketimAnalizModeli:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=10)
        self.model_egitildi = False
    
    def veri_on_isleme(self, df):
        """Veriyi analiz için hazırlar"""
        # Tarih dönüşümleri
        df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
        df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
        
        # Temel özellik mühendisliği
        df['OKUMA_PERIYODU_GUN'] = (df['OKUMA_TARIHI'] - df['ILK_OKUMA_TARIHI']).dt.days
        df['OKUMA_PERIYODU_GUN'] = df['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        df['GUNLUK_ORT_TUKETIM_m3'] = df['AKTIF_m3'] / df['OKUMA_PERIYODU_GUN']
        df['GUNLUK_ORT_TUKETIM_m3'] = df['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        
        # Ek özellikler
        df['TUKEIM_VARYASYON'] = df.groupby('TESISAT_NO')['AKTIF_m3'].transform('std')
        df['TUKEIM_ORTALAMA'] = df.groupby('TESISAT_NO')['AKTIF_m3'].transform('mean')
        df['VARYASYON_KATSAYISI'] = df['TUKEIM_VARYASYON'] / df['TUKEIM_ORTALAMA']
        df['VARYASYON_KATSAYISI'] = df['VARYASYON_KATSAYISI'].fillna(0)
        
        return df
    
    def gelismis_davranis_analizi(self, tesisat_verisi):
        """Gelişmiş davranış analizi yapar"""
        if len(tesisat_verisi) < 3:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta", 0
        
        tuketimler = tesisat_verisi['AKTIF_m3'].values
        tarihler = tesisat_verisi['OKUMA_TARIHI']
        
        # İstatistiksel özellikler
        sifir_sayisi = sum(tuketimler == 0)
        sifir_orani = sifir_sayisi / len(tuketimler)
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0
        
        # Trend analizi
        if len(tuketimler) >= 5:
            son_bes = tuketimler[-5:]
            trend_degeri = (son_bes[-1] - son_bes[0]) / son_bes[0] if son_bes[0] > 0 else 0
        elif len(tuketimler) >= 3:
            son_uc = tuketimler[-3:]
            trend_degeri = (son_uc[-1] - son_uc[0]) / son_uc[0] if son_uc[0] > 0 else 0
        else:
            trend_degeri = 0
        
        # Risk puanı hesaplama
        risk_puan = 0
        
        # 1. Sıfır tüketim analizi
        if sifir_sayisi >= 2:
            sifir_indisler = np.where(tuketimler == 0)[0]
            if len(sifir_indisler) >= 2:
                ardisik_olmayan = sum(np.diff(sifir_indisler) > 1) >= 1
                if ardisik_olmayan:
                    risk_puan += 3
        
        if sifir_orani > 0.5:
            risk_puan += 2
        
        # 2. Varyasyon analizi
        if varyasyon_katsayisi > 1.5:
            risk_puan += 2
        elif varyasyon_katsayisi > 1.0:
            risk_puan += 1
        
        # 3. Trend analizi
        if abs(trend_degeri) > 0.3:
            risk_puan += 2
        elif abs(trend_degeri) > 0.1:
            risk_puan += 1
        
        # 4. Son dönem sıfır tüketim
        if tuketimler[-1] == 0 and len(tuketimler) > 1:
            risk_puan += 2
        
        # 5. Anormal yüksek tüketim
        if mean_tuketim > 50:
            risk_puan += 2
        elif mean_tuketim > 20:
            risk_puan += 1
        
        # Risk seviyesi belirleme
        if risk_puan >= 5:
            risk_seviyesi = "Yüksek"
        elif risk_puan >= 3:
            risk_seviyesi = "Orta"
        else:
            risk_seviyesi = "Düşük"
        
        # Şüpheli dönem tespiti
        supheli_donemler = []
        if sifir_sayisi > 0:
            for idx in np.where(tuketimler == 0)[0]:
                tarih_obj = pd.Timestamp(tarihler.iloc[idx])
                supheli_donemler.append(tarih_obj.strftime('%m/%Y'))
        
        # Yorum oluşturma
        if risk_seviyesi == "Yüksek":
            yorum = self._yuksek_risk_yorumu_olustur(tuketimler, sifir_sayisi, varyasyon_katsayisi, trend_degeri)
        elif risk_seviyesi == "Orta":
            yorum = self._orta_risk_yorumu_olustur(tuketimler, sifir_sayisi, varyasyon_katsayisi)
        else:
            yorum = self._dusuk_risk_yorumu_olustur()
        
        return yorum, ", ".join(supheli_donemler) if supheli_donemler else "Yok", risk_seviyesi, risk_puan
    
    def _yuksek_risk_yorumu_olustur(self, tuketimler, sifir_sayisi, varyasyon_katsayisi, trend_degeri):
        """Yüksek risk için yorum oluşturur"""
        yorumlar = []
        
        if sifir_sayisi >= 2:
            yorumlar.append("Düzensiz sıfır tüketim paterni")
        if varyasyon_katsayisi > 1.5:
            yorumlar.append("Yüksek tüketim dalgalanması")
        if abs(trend_degeri) > 0.3:
            yorumlar.append(f"{'Yükselen' if trend_degeri > 0 else 'Düşen'} tüketim trendi")
        if np.mean(tuketimler) > 50:
            yorumlar.append("Anormal yüksek tüketim")
        
        return " | ".join(yorumlar) + " - Acil inceleme önerilir"
    
    def _orta_risk_yorumu_olustur(self, tuketimler, sifir_sayisi, varyasyon_katsayisi):
        """Orta risk için yorum oluşturur"""
        if sifir_sayisi == 1:
            return "Tekil sıfır tüketim - İzleme gerektirir"
        elif varyasyon_katsayisi > 1.0:
            return "Orta seviyede tüketim dalgalanması"
        else:
            return "Tüketim davranışında küçük tutarsızlıklar"
    
    def _dusuk_risk_yorumu_olustur(self):
        """Düşük risk için yorum oluşturur"""
        yorumlar = [
            "Normal tüketim paterni",
            "Stabil tüketim alışkanlığı",
            "Tutarlı tüketim davranışı"
        ]
        return np.random.choice(yorumlar)
    
    def anomaly_detection(self, df):
        """Anomali tespiti yapar"""
        features = df[['AKTIF_m3', 'GUNLUK_ORT_TUKETIM_m3', 'VARYASYON_KATSAYISI']].fillna(0)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Isolation Forest ile anomali tespiti
        anomalies = self.isolation_forest.fit_predict(features_scaled)
        df['ANOMALY_SCORE'] = anomalies
        
        return df
    
    def save_model(self, filepath):
        """Modeli kaydeder"""
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Modeli yükler"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.model_egitildi = True

# Global model instance
analiz_modeli = SuTuketimAnalizModeli()
