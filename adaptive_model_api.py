import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import joblib

warnings.filterwarnings('ignore')

class AdaptiveSuTuketimModeli:
    def __init__(self, model_path="adaptive_model.joblib"):
        self.model_path = model_path
        self.learning_data = []
        self.adaptive_thresholds = {
            'varyasyon_esik': 1.5,
            'yuksek_tuketim_esik': 50,
            'trend_esik': 0.3,
            'sifir_esik': 2
        }
        self.pattern_memory = {}
        self.performance_history = []
        self.load_model()
    
    def load_model(self):
        """Öğrenilmiş modeli yükler"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.adaptive_thresholds = model_data.get('adaptive_thresholds', self.adaptive_thresholds)
                self.pattern_memory = model_data.get('pattern_memory', {})
                self.performance_history = model_data.get('performance_history', [])
                print("✅ Öğrenilmiş model yüklendi")
        except Exception as e:
            print(f"❌ Model yüklenemedi: {e}")
    
    def save_model(self):
        """Modeli kaydeder"""
        try:
            model_data = {
                'adaptive_thresholds': self.adaptive_thresholds,
                'pattern_memory': self.pattern_memory,
                'performance_history': self.performance_history,
                'last_update': datetime.now()
            }
            joblib.dump(model_data, self.model_path)
            print("✅ Model kaydedildi")
        except Exception as e:
            print(f"❌ Model kaydedilemedi: {e}")
    
    def learn_from_feedback(self, tesisat_no, gercek_durum, tahmin_durum):
        """Geri bildirimle öğrenme"""
        feedback = {
            'tesisat_no': tesisat_no,
            'gercek_durum': gercek_durum,
            'tahmin_durum': tahmin_durum,
            'tarih': datetime.now(),
            'basari': 1 if gercek_durum == tahmin_durum else 0
        }
        
        self.performance_history.append(feedback)
        
        # Son 1000 kaydı tut (bellek şişmesin)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Başarı oranına göre threshold'ları ayarla
        self.adaptive_learning()
        self.save_model()
    
    def adaptive_learning(self):
        """Adaptif öğrenme mekanizması"""
        if len(self.performance_history) < 50:
            return
        
        son_performans = self.performance_history[-100:]
        basari_orani = sum([p['basari'] for p in son_performans]) / len(son_performans)
        
        # Başarı oranına göre threshold'ları optimize et
        if basari_orani < 0.7:  # Başarı düşükse threshold'ları sıkılaştır
            self.adaptive_thresholds['varyasyon_esik'] *= 0.9
            self.adaptive_thresholds['trend_esik'] *= 0.9
        elif basari_orani > 0.9:  # Başarı yüksekse threshold'ları gevşet
            self.adaptive_thresholds['varyasyon_esik'] *= 1.1
            self.adaptive_thresholds['trend_esik'] *= 1.1
        
        # Threshold'ları makul sınırlarda tut
        self.adaptive_thresholds['varyasyon_esik'] = max(0.5, min(3.0, self.adaptive_thresholds['varyasyon_esik']))
        self.adaptive_thresholds['trend_esik'] = max(0.1, min(1.0, self.adaptive_thresholds['trend_esik']))
    
    def incremental_analysis(self, yeni_veri, onceki_analiz=None):
        """Artımsal analiz - tüm veriyi yeniden işlemez"""
        if onceki_analiz is None:
            return self.gelismis_davranis_analizi(yeni_veri)
        
        # Önceki analizi güncelle
        # Sadece yeni pattern'leri kontrol et
        guncel_tuketimler = yeni_veri['AKTIF_m3'].values
        
        # Pattern değişikliği var mı?
        pattern_degisiklik = self.detect_pattern_change(onceki_analiz, guncel_tuketimler)
        
        if pattern_degisiklik:
            return self.gelismis_davranis_analizi(yeni_veri)
        else:
            # Pattern değişmediyse önceki analizi döndür
            return onceki_analiz
    
    def detect_pattern_change(self, onceki_analiz, yeni_tuketimler):
        """Pattern değişikliğini tespit et"""
        if len(yeni_tuketimler) < 3:
            return False
        
        # Basit pattern değişiklik kontrolü
        onceki_std = onceki_analiz.get('std_dev', 0)
        yeni_std = np.std(yeni_tuketimler)
        
        onceki_mean = onceki_analiz.get('mean_tuketim', 0)
        yeni_mean = np.mean(yeni_tuketimler)
        
        # Önemli değişiklik var mı?
        std_degisim = abs(yeni_std - onceki_std) / max(onceki_std, 0.001)
        mean_degisim = abs(yeni_mean - onceki_mean) / max(onceki_mean, 0.001)
        
        return std_degisim > 0.5 or mean_degisim > 0.3
    
    def gelismis_davranis_analizi(self, tesisat_verisi):
        """Gelişmiş davranış analizi - adaptive threshold'larla"""
        if len(tesisat_verisi) < 3:
            return {
                'yorum': "Yetersiz veri",
                'supheli_donemler': "Yetersiz kayıt", 
                'risk_seviyesi': "Orta",
                'risk_puan': 0,
                'std_dev': 0,
                'mean_tuketim': 0
            }
        
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
        
        # ADAPTIVE risk puanı hesaplama
        risk_puan = 0
        
        # 1. Sıfır tüketim analizi - adaptive threshold
        sifir_esik = self.adaptive_thresholds['sifir_esik']
        if sifir_sayisi >= sifir_esik:
            risk_puan += 3
        elif sifir_sayisi == 1:
            risk_puan += 1
        
        if sifir_orani > 0.5:
            risk_puan += 2
        
        # 2. Varyasyon analizi - adaptive threshold
        varyasyon_esik = self.adaptive_thresholds['varyasyon_esik']
        if varyasyon_katsayisi > varyasyon_esik:
            risk_puan += 2
        elif varyasyon_katsayisi > varyasyon_esik * 0.7:
            risk_puan += 1
        
        # 3. Trend analizi - adaptive threshold  
        trend_esik = self.adaptive_thresholds['trend_esik']
        if abs(trend_degeri) > trend_esik:
            risk_puan += 2
        elif abs(trend_degeri) > trend_esik * 0.7:
            risk_puan += 1
        
        # 4. Son dönem sıfır tüketim
        if len(tuketimler) > 1 and tuketimler[-1] == 0:
            risk_puan += 2
        
        # 5. Anormal yüksek tüketim - adaptive threshold
        yuksek_tuketim_esik = self.adaptive_thresholds['yuksek_tuketim_esik']
        if mean_tuketim > yuksek_tuketim_esik:
            risk_puan += 2
        elif mean_tuketim > yuksek_tuketim_esik * 0.7:
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
        yorum = self._adaptive_yorum_olustur(risk_seviyesi, risk_puan, 
                                           sifir_sayisi, varyasyon_katsayisi, 
                                           trend_degeri, mean_tuketim)
        
        return {
            'yorum': yorum,
            'supheli_donemler': ", ".join(supheli_donemler) if supheli_donemler else "Yok",
            'risk_seviyesi': risk_seviyesi,
            'risk_puan': risk_puan,
            'std_dev': std_dev,
            'mean_tuketim': mean_tuketim
        }
    
    def _adaptive_yorum_olustur(self, risk_seviyesi, risk_puan, sifir_sayisi, 
                              varyasyon_katsayisi, trend_degeri, mean_tuketim):
        """Adaptive yorum oluşturma"""
        
        # Öğrenilmiş pattern'lere göre yorum
        if risk_seviyesi == "Yüksek":
            yorumlar = []
            varyasyon_esik = self.adaptive_thresholds['varyasyon_esik']
            trend_esik = self.adaptive_thresholds['trend_esik']
            yuksek_esik = self.adaptive_thresholds['yuksek_tuketim_esik']
            
            if sifir_sayisi >= self.adaptive_thresholds['sifir_esik']:
                yorumlar.append("Düzensiz sıfır tüketim paterni")
            if varyasyon_katsayisi > varyasyon_esik:
                yorumlar.append("Yüksek tüketim dalgalanması")
            if abs(trend_degeri) > trend_esik:
                yorumlar.append(f"{'Yükselen' if trend_degeri > 0 else 'Düşen'} tüketim trendi")
            if mean_tuketim > yuksek_esik:
                yorumlar.append("Anormal yüksek tüketim")
            
            if yorumlar:
                return " | ".join(yorumlar) + " - Acil inceleme önerilir"
            else:
                return "Yüksek riskli tüketim paterni - İnceleme gerekli"
        
        elif risk_seviyesi == "Orta":
            if sifir_sayisi == 1:
                return "Tekil sıfır tüketim - İzleme gerektirir"
            elif varyasyon_katsayisi > self.adaptive_thresholds['varyasyon_esik'] * 0.7:
                return "Orta seviyede tüketim dalgalanması"
            else:
                return "Tüketim davranışında küçük tutarsızlıklar"
        
        else:
            yorumlar = [
                "Normal tüketim paterni",
                "Stabil tüketim alışkanlığı", 
                "Tutarlı tüketim davranışı"
            ]
            return np.random.choice(yorumlar)
    
    def get_learning_stats(self):
        """Öğrenme istatistiklerini getir"""
        if not self.performance_history:
            return {
                'toplam_gozlem': 0,
                'basari_orani': 0,
                'adaptive_thresholds': self.adaptive_thresholds
            }
        
        toplam_gozlem = len(self.performance_history)
        basari_orani = sum([p['basari'] for p in self.performance_history]) / toplam_gozlem
        
        return {
            'toplam_gozlem': toplam_gozlem,
            'basari_orani': basari_orani,
            'adaptive_thresholds': self.adaptive_thresholds
        }

# Global adaptive model instance
adaptive_model = AdaptiveSuTuketimModeli()