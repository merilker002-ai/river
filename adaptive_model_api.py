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
        
        # DAHA AKILLI BAŞLANGIÇ THRESHOLD'LARI
        self.adaptive_thresholds = {
            'varyasyon_esik': 1.2,    # Daha hassas başla
            'yuksek_tuketim_esik': 40, # Daha düşük başla
            'trend_esik': 0.25,       # Daha hassas trend
            'sifir_esik': 1           # Daha hassas sıfır tespiti
        }
        
        self.pattern_memory = {}
        self.performance_history = []
        
        # OTOMATİK ÖĞRENME VERİSİ
        self._initialize_with_synthetic_data()
        self.load_model()
    
    def _initialize_with_synthetic_data(self):
        """Sentetik veri ile hemen öğrenmeye başla"""
        print("🤖 Sentetik veri ile AI eğitiliyor...")
        
        # Başarılı tahminler (gerçek hayattan beklenen pattern'ler)
        successful_patterns = [
            # Normal pattern'ler - Düşük risk
            {'sifir_sayisi': 0, 'varyasyon': 0.5, 'trend': 0.05, 'tuketim': 15, 'risk': 'Düşük'},
            {'sifir_sayisi': 0, 'varyasyon': 0.8, 'trend': 0.08, 'tuketim': 25, 'risk': 'Düşük'},
            
            # Orta risk pattern'leri
            {'sifir_sayisi': 1, 'varyasyon': 1.1, 'trend': 0.15, 'tuketim': 35, 'risk': 'Orta'},
            {'sifir_sayisi': 0, 'varyasyon': 1.4, 'trend': 0.12, 'tuketim': 45, 'risk': 'Orta'},
            
            # Yüksek risk pattern'leri
            {'sifir_sayisi': 2, 'varyasyon': 1.8, 'trend': 0.35, 'tuketim': 60, 'risk': 'Yüksek'},
            {'sifir_sayisi': 3, 'varyasyon': 2.2, 'trend': 0.45, 'tuketim': 80, 'risk': 'Yüksek'},
        ]
        
        # Sentetik feedback'ler oluştur
        for pattern in successful_patterns:
            feedback = {
                'tesisat_no': f"SYNTHETIC_{hash(str(pattern))}",
                'gercek_durum': pattern['risk'],
                'tahmin_durum': pattern['risk'],  # Doğru tahmin
                'tarih': datetime.now(),
                'basari': 1,
                'pattern': pattern
            }
            self.performance_history.append(feedback)
        
        print(f"✅ {len(successful_patterns)} sentetik pattern ile AI eğitildi")
    
    def load_model(self):
        """Öğrenilmiş modeli yükler - daha güçlü hata yönetimi"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.adaptive_thresholds = model_data.get('adaptive_thresholds', self.adaptive_thresholds)
                self.pattern_memory = model_data.get('pattern_memory', {})
                
                # Mevcut performans geçmişine ekle (çakışma olmasın)
                existing_history = model_data.get('performance_history', [])
                existing_ids = [p.get('tesisat_no') for p in self.performance_history]
                
                for item in existing_history:
                    if item.get('tesisat_no') not in existing_ids:
                        self.performance_history.append(item)
                
                print(f"✅ Öğrenilmiş model yüklendi. Toplam gözlem: {len(self.performance_history)}")
                
                # Threshold'ları optimize et
                self.adaptive_learning()
                
        except Exception as e:
            print(f"❌ Model yüklenemedi, sentetik veri ile devam: {e}")
    
    def save_model(self):
        """Modeli kaydeder - daha güvenli"""
        try:
            model_data = {
                'adaptive_thresholds': self.adaptive_thresholds,
                'pattern_memory': self.pattern_memory,
                'performance_history': self.performance_history[-2000:],  # Bellek optimizasyonu
                'last_update': datetime.now(),
                'version': '1.1',
                'total_observations': len(self.performance_history)
            }
            joblib.dump(model_data, self.model_path)
            print(f"✅ Model kaydedildi. Toplam gözlem: {len(self.performance_history)}")
        except Exception as e:
            print(f"❌ Model kaydedilemedi: {e}")
    
    def learn_from_feedback(self, tesisat_no, gercek_durum, tahmin_durum, pattern_data=None):
        """Gelişmiş geri bildirimle öğrenme"""
        feedback = {
            'tesisat_no': tesisat_no,
            'gercek_durum': gercek_durum,
            'tahmin_durum': tahmin_durum,
            'tarih': datetime.now(),
            'basari': 1 if gercek_durum == tahmin_durum else 0,
            'pattern': pattern_data
        }
        
        # Benzersiz feedback'leri ekle
        existing_ids = [p.get('tesisat_no') for p in self.performance_history]
        if tesisat_no not in existing_ids:
            self.performance_history.append(feedback)
        
        # Performansı hemen güncelle
        self.adaptive_learning()
        self.save_model()
        
        print(f"📝 Yeni feedback: {tesisat_no} | Gerçek: {gercek_durum} | Tahmin: {tahmin_durum} | Başarı: {feedback['basari']}")
    
    def adaptive_learning(self):
        """Daha agresif adaptif öğrenme"""
        if len(self.performance_history) < 10:
            return
        
        # Son 500 kaydı değerlendir
        evaluation_data = self.performance_history[-500:] if len(self.performance_history) > 500 else self.performance_history
        basari_orani = sum([p['basari'] for p in evaluation_data]) / len(evaluation_data)
        
        print(f"🎯 Öğrenme Değerlendirmesi: {len(evaluation_data)} gözlem, Başarı: {basari_orani:.1%}")
        
        # DAHA HIZLI ÖĞRENME
        learning_rate = 0.1  # Öğrenme hızını artır
        
        if basari_orani < 0.6:  # Başarı düşükse threshold'ları optimize et
            self.adaptive_thresholds['varyasyon_esik'] *= (1 - learning_rate)
            self.adaptive_thresholds['trend_esik'] *= (1 - learning_rate)
            self.adaptive_thresholds['yuksek_tuketim_esik'] *= (1 - learning_rate * 0.5)
            print("🔧 Threshold'lar sıkılaştırıldı (düşük başarı)")
            
        elif basari_orani > 0.85:  # Başarı yüksekse threshold'ları gevşet
            self.adaptive_thresholds['varyasyon_esik'] *= (1 + learning_rate)
            self.adaptive_thresholds['trend_esik'] *= (1 + learning_rate)
            self.adaptive_thresholds['yuksek_tuketim_esik'] *= (1 + learning_rate * 0.5)
            print("🔧 Threshold'lar gevşetildi (yüksek başarı)")
        
        # Threshold'ları makul sınırlarda tut
        self.adaptive_thresholds['varyasyon_esik'] = max(0.3, min(3.0, self.adaptive_thresholds['varyasyon_esik']))
        self.adaptive_thresholds['trend_esik'] = max(0.05, min(1.0, self.adaptive_thresholds['trend_esik']))
        self.adaptive_thresholds['yuksek_tuketim_esik'] = max(10, min(200, self.adaptive_thresholds['yuksek_tuketim_esik']))
        self.adaptive_thresholds['sifir_esik'] = max(1, min(5, self.adaptive_thresholds['sifir_esik']))
        
        print(f"📊 Yeni Threshold'lar: {self.adaptive_thresholds}")
    
    def auto_learn_from_analysis(self, tesisat_verisi, analiz_sonucu):
        """Analiz sonuçlarından otomatik öğrenme"""
        if len(tesisat_verisi) < 6:  # Yeterli veri yoksa öğrenme
            return
        
        tuketimler = tesisat_verisi['AKTIF_m3'].values
        
        # Pattern verilerini topla
        pattern_data = {
            'sifir_sayisi': sum(tuketimler == 0),
            'varyasyon': np.std(tuketimler) / np.mean(tuketimler) if np.mean(tuketimler) > 0 else 0,
            'trend': self._calculate_trend(tuketimler),
            'tuketim': np.mean(tuketimler),
            'length': len(tuketimler)
        }
        
        # Pattern'i hafızaya kaydet
        pattern_key = f"pattern_{hash(str(pattern_data))}"
        self.pattern_memory[pattern_key] = {
            'pattern': pattern_data,
            'risk_seviyesi': analiz_sonucu['risk_seviyesi'],
            'count': self.pattern_memory.get(pattern_key, {}).get('count', 0) + 1,
            'last_seen': datetime.now()
        }
    
    def _calculate_trend(self, tuketimler):
        """Trend hesaplama"""
        if len(tuketimler) < 3:
            return 0
        return (tuketimler[-1] - tuketimler[0]) / tuketimler[0] if tuketimler[0] > 0 else 0

    def gelismis_davranis_analizi(self, tesisat_verisi):
        """Gelişmiş davranış analizi - öğrenme entegre"""
        if len(tesisat_verisi) < 3:
            return self._create_default_analysis("Yetersiz veri", "Orta", 0)
        
        tuketimler = tesisat_verisi['AKTIF_m3'].values
        tarihler = tesisat_verisi['OKUMA_TARIHI']
        
        # İstatistiksel özellikler
        sifir_sayisi = sum(tuketimler == 0)
        sifir_orani = sifir_sayisi / len(tuketimler)
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0
        
        # Trend analizi
        trend_degeri = self._calculate_trend(tuketimler)
        
        # ÖĞRENİLMİŞ THRESHOLD'LAR ile risk puanı hesaplama
        risk_puan = self._calculate_adaptive_risk_score(
            sifir_sayisi, sifir_orani, varyasyon_katsayisi, 
            trend_degeri, mean_tuketim, len(tuketimler)
        )
        
        # Risk seviyesi belirleme
        risk_seviyesi = self._determine_risk_level(risk_puan)
        
        # Şüpheli dönem tespiti
        supheli_donemler = self._find_suspicious_periods(tuketimler, tarihler, sifir_sayisi)
        
        # ÖĞRENİLMİŞ YORUM oluşturma
        yorum = self._adaptive_yorum_olustur(
            risk_seviyesi, risk_puan, sifir_sayisi, 
            varyasyon_katsayisi, trend_degeri, mean_tuketim
        )
        
        # OTOMATİK ÖĞRENME
        self.auto_learn_from_analysis(tesisat_verisi, {
            'risk_seviyesi': risk_seviyesi,
            'risk_puan': risk_puan,
            'yorum': yorum
        })
        
        return {
            'yorum': yorum,
            'supheli_donemler': supheli_donemler,
            'risk_seviyesi': risk_seviyesi,
            'risk_puan': risk_puan,
            'std_dev': std_dev,
            'mean_tuketim': mean_tuketim,
            'pattern_data': {
                'sifir_sayisi': sifir_sayisi,
                'varyasyon_katsayisi': varyasyon_katsayisi,
                'trend_degeri': trend_degeri,
                'mean_tuketim': mean_tuketim
            }
        }
    
    def _calculate_adaptive_risk_score(self, sifir_sayisi, sifir_orani, varyasyon_katsayisi, trend_degeri, mean_tuketim, data_length):
        """Adaptive risk skoru hesaplama"""
        risk_puan = 0
        
        # 1. Sıfır tüketim analizi - adaptive threshold
        if sifir_sayisi >= self.adaptive_thresholds['sifir_esik']:
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
        
        # 4. Son dönem sıfır tüketim (basit kontrol)
        # Bu kısım tesisat_verisi gerektirdiği için ana fonksiyonda yapılıyor
        
        # 5. Anormal yüksek tüketim - adaptive threshold
        yuksek_tuketim_esik = self.adaptive_thresholds['yuksek_tuketim_esik']
        if mean_tuketim > yuksek_tuketim_esik:
            risk_puan += 2
        elif mean_tuketim > yuksek_tuketim_esik * 0.7:
            risk_puan += 1
        
        return risk_puan
    
    def _determine_risk_level(self, risk_puan):
        """Risk seviyesi belirleme"""
        if risk_puan >= 5:
            return "Yüksek"
        elif risk_puan >= 3:
            return "Orta"
        else:
            return "Düşük"
    
    def _find_suspicious_periods(self, tuketimler, tarihler, sifir_sayisi):
        """Şüpheli dönemleri bul"""
        supheli_donemler = []
        if sifir_sayisi > 0:
            for idx in np.where(tuketimler == 0)[0]:
                if idx < len(tarihler):
                    try:
                        tarih_obj = pd.Timestamp(tarihler.iloc[idx])
                        supheli_donemler.append(tarih_obj.strftime('%m/%Y'))
                    except:
                        continue
        return ", ".join(supheli_donemler) if supheli_donemler else "Yok"
    
    def _adaptive_yorum_olustur(self, risk_seviyesi, risk_puan, sifir_sayisi, varyasyon_katsayisi, trend_degeri, mean_tuketim):
        """Adaptive yorum oluşturma"""
        
        if risk_seviyesi == "Yüksek":
            yorumlar = []
            
            if sifir_sayisi >= self.adaptive_thresholds['sifir_esik']:
                yorumlar.append("Düzensiz sıfır tüketim paterni")
            if varyasyon_katsayisi > self.adaptive_thresholds['varyasyon_esik']:
                yorumlar.append("Yüksek tüketim dalgalanması")
            if abs(trend_degeri) > self.adaptive_thresholds['trend_esik']:
                yorumlar.append(f"{'Yükselen' if trend_degeri > 0 else 'Düşen'} tüketim trendi")
            if mean_tuketim > self.adaptive_thresholds['yuksek_tuketim_esik']:
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
    
    def _create_default_analysis(self, yorum, risk_seviyesi, risk_puan):
        """Varsayılan analiz oluştur"""
        return {
            'yorum': yorum,
            'supheli_donemler': "Yok",
            'risk_seviyesi': risk_seviyesi,
            'risk_puan': risk_puan,
            'std_dev': 0,
            'mean_tuketim': 0
        }
    
    def get_learning_stats(self):
        """Detaylı öğrenme istatistiklerini getir"""
        if not self.performance_history:
            return {
                'toplam_gozlem': 0,
                'basari_orani': 0,
                'adaptive_thresholds': self.adaptive_thresholds,
                'model_version': '1.1',
                'status': 'Sentetik veri ile başlatıldı'
            }
        
        toplam_gozlem = len(self.performance_history)
        
        # Gerçek feedback'leri filtrele (sentetik olmayanlar)
        real_feedbacks = [p for p in self.performance_history if not p.get('tesisat_no', '').startswith('SYNTHETIC_')]
        real_basari_orani = sum([p['basari'] for p in real_feedbacks]) / len(real_feedbacks) if real_feedbacks else 0
        
        # Tüm feedback'ler
        total_basari_orani = sum([p['basari'] for p in self.performance_history]) / toplam_gozlem
        
        return {
            'toplam_gozlem': toplam_gozlem,
            'gercek_gozlem': len(real_feedbacks),
            'basari_orani': total_basari_orani,
            'gercek_basari_orani': real_basari_orani,
            'adaptive_thresholds': self.adaptive_thresholds,
            'model_version': '1.1',
            'status': 'Aktif öğrenme modunda',
            'pattern_memory_size': len(self.pattern_memory)
        }

# Global adaptive model instance
adaptive_model = AdaptiveSuTuketimModeli()
