import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SuTuketimAnalizModeli:
    def __init__(self):
        self.model_egitildi = False
    
    def veri_on_isleme(self, df):
        """Veriyi analiz için hazırlar"""
        # Tarih dönüşümleri
        df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
        df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
        
        # Sayısal sütunları temizle
        numeric_columns = ['AKTIF_m3', 'TOPLAM_TUTAR']
        for col in numeric_columns:
            if col in df.columns:
                # String değerleri temizle ve sayısal yap
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # NaN değerleri 0 yap
                df[col] = df[col].fillna(0)
        
        # Temel özellik mühendisliği
        df['OKUMA_PERIYODU_GUN'] = (df['OKUMA_TARIHI'] - df['ILK_OKUMA_TARIHI']).dt.days
        df['OKUMA_PERIYODU_GUN'] = df['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        df['GUNLUK_ORT_TUKETIM_m3'] = df['AKTIF_m3'] / df['OKUMA_PERIYODU_GUN']
        df['GUNLUK_ORT_TUKETIM_m3'] = df['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        
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
            risk_puan += 3
        elif sifir_sayisi == 1:
            risk_puan += 1
        
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
        if len(tuketimler) > 1 and tuketimler[-1] == 0:
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
                if idx < len(tarihler):
                    try:
                        tarih_obj = pd.Timestamp(tarihler.iloc[idx])
                        supheli_donemler.append(tarih_obj.strftime('%m/%Y'))
                    except:
                        continue
        
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
        
        if yorumlar:
            return " | ".join(yorumlar) + " - Acil inceleme önerilir"
        else:
            return "Yüksek riskli tüketim paterni - İnceleme gerekli"
    
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
        """Basit anomali tespiti yapar"""
        try:
            # IQR (Interquartile Range) yöntemi ile anomali tespiti
            Q1 = df['AKTIF_m3'].quantile(0.25)
            Q3 = df['AKTIF_m3'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Anomalileri işaretle
            df['ANOMALY_SCORE'] = np.where(
                (df['AKTIF_m3'] < lower_bound) | (df['AKTIF_m3'] > upper_bound), -1, 1
            )
        except:
            # Hata durumunda tümünü normal olarak işaretle
            df['ANOMALY_SCORE'] = 1
        
        return df

# Global model instance
analiz_modeli = SuTuketimAnalizModeli()
