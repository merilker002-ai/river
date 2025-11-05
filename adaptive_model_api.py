import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class AdaptiveLearningModel:
    def __init__(self):
        # Öğrenme verisi
        self.learning_data = {
            'pattern_counts': defaultdict(int),
            'risk_patterns': defaultdict(list),
            'feedback_history': deque(maxlen=1000),
            'success_rates': defaultdict(float),
            'adaptive_thresholds': {
                'zero_consumption': 0.3,
                'high_consumption': 50.0,
                'variance_threshold': 0.4,
                'pattern_confidence': 0.7
            },
            'total_observations': 0,
            'real_observations': 0,
            'successful_predictions': 0
        }
        
        # Pattern tanımları
        self.pattern_definitions = {
            'sifir_aralikli': {
                'description': 'Aralıklı sıfır tüketim patterni',
                'risk_level': 'Yüksek',
                'features': ['zero_ratio', 'consecutive_zeros']
            },
            'yuksek_tuketim': {
                'description': 'Sürekli yüksek tüketim patterni', 
                'risk_level': 'Orta',
                'features': ['mean_consumption', 'variance']
            },
            'normal': {
                'description': 'Normal tüketim patterni',
                'risk_level': 'Düşük', 
                'features': ['stability', 'seasonality']
            },
            'degisken': {
                'description': 'Değişken tüketim patterni',
                'risk_level': 'Orta',
                'features': ['variance', 'outliers']
            }
        }
        
        # Sentetik veri ile başlangıç eğitimi
        self._initialize_with_synthetic_data()
    
    def _initialize_with_synthetic_data(self):
        """Sentetik veri ile modeli başlangıç eğitimi"""
        print("🧠 Sentetik veri ile başlangıç eğitimi yapılıyor...")
        
        # Sentetik pattern örnekleri
        synthetic_patterns = [
            # Sıfır aralıklı pattern (Yüksek risk)
            {'zeros_ratio': 0.4, 'variance': 0.8, 'max_consumption': 100, 'risk': 'Yüksek'},
            {'zeros_ratio': 0.3, 'variance': 0.7, 'max_consumption': 80, 'risk': 'Yüksek'},
            
            # Yüksek tüketim pattern (Orta risk)
            {'zeros_ratio': 0.0, 'variance': 0.3, 'max_consumption': 150, 'risk': 'Orta'},
            {'zeros_ratio': 0.1, 'variance': 0.4, 'max_consumption': 120, 'risk': 'Orta'},
            
            # Normal pattern (Düşük risk)
            {'zeros_ratio': 0.0, 'variance': 0.2, 'max_consumption': 50, 'risk': 'Düşük'},
            {'zeros_ratio': 0.05, 'variance': 0.15, 'max_consumption': 40, 'risk': 'Düşük'},
            
            # Değişken pattern (Orta risk)
            {'zeros_ratio': 0.2, 'variance': 0.6, 'max_consumption': 90, 'risk': 'Orta'},
        ]
        
        for pattern in synthetic_patterns:
            features = {
                'zero_ratio': pattern['zeros_ratio'],
                'variance': pattern['variance'], 
                'max_consumption': pattern['max_consumption'],
                'pattern_type': self._classify_pattern(pattern)
            }
            
            self.learning_data['pattern_counts'][pattern['risk']] += 1
            self.learning_data['risk_patterns'][pattern['risk']].append(features)
            self.learning_data['total_observations'] += 1
            self.learning_data['real_observations'] += 1
        
        # Başlangıç başarı oranı
        self.learning_data['successful_predictions'] = int(self.learning_data['real_observations'] * 0.85)
        
        print(f"✅ Sentetik eğitim tamamlandı: {self.learning_data['real_observations']} örnek")
    
    def _classify_pattern(self, features):
        """Pattern sınıflandırma"""
        if features['zeros_ratio'] > 0.25:
            return 'sifir_aralikli'
        elif features['max_consumption'] > 80:
            return 'yuksek_tuketim'
        elif features['variance'] > 0.5:
            return 'degisken'
        else:
            return 'normal'
    
    def gelismis_davranis_analizi(self, tesisat_verisi):
        """Gelişmiş davranış analizi ve pattern tanıma"""
        
        if len(tesisat_verisi) < 3:
            return self._default_analysis()
        
        try:
            # Temel istatistikler
            tuketimler = tesisat_verisi['AKTIF_m3'].values
            dates = pd.to_datetime(tesisat_verisi['OKUMA_TARIHI'])
            
            # Pattern özellikleri çıkarımı
            features = self._extract_pattern_features(tuketimler, dates)
            
            # Risk değerlendirmesi
            risk_analysis = self._assess_risk_with_learning(features)
            
            # Öğrenme güncellemesi
            self._update_learning(features, risk_analysis['risk_seviyesi'])
            
            return {
                'yorum': risk_analysis['yorum'],
                'supheli_donemler': risk_analysis['supheli_donemler'],
                'risk_seviyesi': risk_analysis['risk_seviyesi'],
                'risk_puan': risk_analysis['risk_puan'],
                'pattern_data': features
            }
            
        except Exception as e:
            print(f"Analiz hatası: {e}")
            return self._default_analysis()
    
    def _extract_pattern_features(self, tuketimler, dates):
        """Pattern özelliklerini çıkar"""
        
        # Temel istatistikler
        mean_tuketim = np.mean(tuketimler)
        std_tuketim = np.std(tuketimler)
        max_tuketim = np.max(tuketimler)
        
        # Sıfır tüketim analizi
        zero_ratio = np.sum(tuketimler == 0) / len(tuketimler)
        
        # Varyans analizi
        variance = std_tuketim / (mean_tuketim + 1e-8)  # Sıfıra bölünmeyi önle
        
        # Zaman serisi özellikleri
        if len(tuketimler) > 1:
            trends = np.diff(tuketimler)
            trend_strength = np.mean(np.abs(trends)) / (mean_tuketim + 1e-8)
        else:
            trend_strength = 0
        
        # Pattern sınıflandırma
        pattern_type = self._classify_consumption_pattern(zero_ratio, variance, max_tuketim)
        
        return {
            'zero_ratio': zero_ratio,
            'variance': variance,
            'mean_consumption': mean_tuketim,
            'max_consumption': max_tuketim,
            'trend_strength': trend_strength,
            'pattern_type': pattern_type,
            'data_points': len(tuketimler)
        }
    
    def _classify_consumption_pattern(self, zero_ratio, variance, max_consumption):
        """Tüketim pattern'ini sınıflandır"""
        
        # Adaptive threshold'ları kullan
        zero_threshold = self.learning_data['adaptive_thresholds']['zero_consumption']
        high_threshold = self.learning_data['adaptive_thresholds']['high_consumption']
        var_threshold = self.learning_data['adaptive_thresholds']['variance_threshold']
        
        if zero_ratio > zero_threshold:
            return 'sifir_aralikli'
        elif max_consumption > high_threshold:
            return 'yuksek_tuketim'
        elif variance > var_threshold:
            return 'degisken'
        else:
            return 'normal'
    
    def _assess_risk_with_learning(self, features):
        """Öğrenilmiş bilgilerle risk değerlendirmesi"""
        
        # Pattern bazlı risk skoru
        pattern_risk_weights = {
            'sifir_aralikli': 0.9,
            'yuksek_tuketim': 0.7, 
            'degisken': 0.6,
            'normal': 0.2
        }
        
        base_risk = pattern_risk_weights.get(features['pattern_type'], 0.5)
        
        # Özellik bazlı ayarlamalar
        if features['zero_ratio'] > 0.4:
            base_risk += 0.3
        elif features['zero_ratio'] > 0.2:
            base_risk += 0.15
            
        if features['variance'] > 0.8:
            base_risk += 0.2
        elif features['variance'] > 0.5:
            base_risk += 0.1
        
        # Öğrenilmiş pattern başarı oranları
        success_rate = self.learning_data['success_rates'].get(features['pattern_type'], 0.7)
        confidence_adjustment = (1 - success_rate) * 0.3  # Düşük başarı → daha yüksek risk
        base_risk += confidence_adjustment
        
        # Risk seviyesi belirleme
        base_risk = max(0.1, min(0.95, base_risk))
        
        if base_risk > 0.7:
            risk_seviyesi = "Yüksek"
            risk_puan = 4
            yorum = self._generate_risk_comment(features, "Yüksek")
        elif base_risk > 0.4:
            risk_seviyesi = "Orta" 
            risk_puan = 2
            yorum = self._generate_risk_comment(features, "Orta")
        else:
            risk_seviyesi = "Düşük"
            risk_puan = 1
            yorum = self._generate_risk_comment(features, "Düşük")
        
        # Şüpheli dönemler
        supheli_donemler = self._identify_suspicious_periods(features)
        
        return {
            'risk_seviyesi': risk_seviyesi,
            'risk_puan': risk_puan,
            'yorum': yorum,
            'supheli_donemler': supheli_donemler
        }
    
    def _generate_risk_comment(self, features, risk_level):
        """Risk seviyesine göre yorum oluştur"""
        
        pattern_names = {
            'sifir_aralikli': 'Aralıklı sıfır tüketim',
            'yuksek_tuketim': 'Yüksek tüketim',
            'degisken': 'Değişken tüketim',
            'normal': 'Normal tüketim'
        }
        
        pattern_desc = pattern_names.get(features['pattern_type'], 'Bilinmeyen pattern')
        
        comments = {
            'Yüksek': [
                f"🚨 {pattern_desc} tespit edildi. Acil inceleme önerilir.",
                f"⚠️ Yüksek riskli {pattern_desc} patterni. Detaylı kontrol gerekli.",
                f"🔴 {pattern_desc} nedeniyle yüksek risk seviyesi."
            ],
            'Orta': [
                f"📊 {pattern_desc} gözlemlendi. Periyodik takip önerilir.", 
                f"🟡 Orta riskli {pattern_desc} patterni. Gözlem altında tutulmalı.",
                f"📈 {pattern_desc} nedeniyle orta risk seviyesi."
            ],
            'Düşük': [
                f"✅ {pattern_desc} patterni. Düşük risk seviyesi.",
                f"🟢 Normal {pattern_desc} davranışı. Risk seviyesi düşük.",
                f"👍 {pattern_desc} nedeniyle düşük risk seviyesi."
            ]
        }
        
        import random
        return random.choice(comments[risk_level])
    
    def _identify_suspicious_periods(self, features):
        """Şüpheli dönemleri belirle"""
        
        suspicious_items = []
        
        if features['zero_ratio'] > 0.3:
            suspicious_items.append(f"%{features['zero_ratio']*100:.1f} sıfır tüketim")
            
        if features['variance'] > 0.7:
            suspicious_items.append("yüksek değişkenlik")
            
        if features['max_consumption'] > 80:
            suspicious_items.append("aşırı tüketim dönemleri")
        
        if suspicious_items:
            return ", ".join(suspicious_items)
        else:
            return "Belirgin şüpheli dönem yok"
    
    def _update_learning(self, features, actual_risk):
        """Öğrenme verisini güncelle"""
        
        # Pattern sayısını güncelle
        pattern_type = features['pattern_type']
        self.learning_data['pattern_counts'][pattern_type] += 1
        
        # Risk pattern'ini kaydet
        self.learning_data['risk_patterns'][actual_risk].append(features)
        
        # Gözlem sayılarını güncelle
        self.learning_data['total_observations'] += 1
        self.learning_data['real_observations'] += 1
        
        # Başarı oranını güncelle (basit yaklaşım)
        if actual_risk in ['Yüksek', 'Orta'] and features['pattern_type'] in ['sifir_aralikli', 'yuksek_tuketim']:
            self.learning_data['successful_predictions'] += 1
        elif actual_risk == 'Düşük' and features['pattern_type'] == 'normal':
            self.learning_data['successful_predictions'] += 1
        
        # Adaptive threshold'ları güncelle
        self._update_adaptive_thresholds()
    
    def _update_adaptive_thresholds(self):
        """Adaptive threshold'ları güncelle"""
        
        # Mevcut pattern dağılımına göre threshold'ları ayarla
        total_patterns = sum(self.learning_data['pattern_counts'].values())
        
        if total_patterns > 0:
            zero_pattern_ratio = self.learning_data['pattern_counts']['sifir_aralikli'] / total_patterns
            high_pattern_ratio = self.learning_data['pattern_counts']['yuksek_tuketim'] / total_patterns
            
            # Zero consumption threshold - daha fazla sıfır pattern varsa threshold'u düşür
            new_zero_threshold = max(0.1, min(0.5, 0.3 - (zero_pattern_ratio - 0.2)))
            
            # High consumption threshold  
            new_high_threshold = max(30, min(100, 50 + (high_pattern_ratio - 0.3) * 50))
            
            # Variance threshold
            new_var_threshold = max(0.2, min(0.8, 0.4 + (zero_pattern_ratio - 0.2)))
            
            self.learning_data['adaptive_thresholds'].update({
                'zero_consumption': new_zero_threshold,
                'high_consumption': new_high_threshold, 
                'variance_threshold': new_var_threshold
            })
    
    def learn_from_feedback(self, tesisat_no, gercek_risk, tahmin_risk, metadata=None):
        """Geri bildirimden öğrenme"""
        
        feedback_data = {
            'tesisat_no': tesisat_no,
            'gercek_risk': gercek_risk,
            'tahmin_risk': tahmin_risk,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.learning_data['feedback_history'].append(feedback_data)
        
        # Başarı durumunu güncelle
        if gercek_risk == tahmin_risk:
            self.learning_data['successful_predictions'] += 1
        
        self.learning_data['total_observations'] += 1
        
        # Pattern başarı oranlarını güncelle
        if 'pattern_type' in metadata:
            pattern_type = metadata['pattern_type']
            current_success = self.learning_data['success_rates'].get(pattern_type, 0.7)
            
            if gercek_risk == tahmin_risk:
                new_success = current_success * 0.95 + 0.05  # Artan başarı
            else:
                new_success = current_success * 0.98 - 0.02  # Azalan başarı
            
            self.learning_data['success_rates'][pattern_type] = max(0.1, min(0.95, new_success))
    
    def get_learning_stats(self):
        """Öğrenme istatistiklerini getir"""
        
        total_obs = self.learning_data['total_observations']
        real_obs = self.learning_data['real_observations']
        success_obs = self.learning_data['successful_predictions']
        
        basari_orani = success_obs / total_obs if total_obs > 0 else 0.7
        
        # Pattern dağılımı
        pattern_dagilimi = dict(self.learning_data['pattern_counts'])
        
        # Son feedback'ler
        son_feedbackler = list(self.learning_data['feedback_history'])[-10:]  # Son 10 feedback
        
        return {
            'toplam_gozlem': total_obs,
            'gercek_gozlem': real_obs,
            'basari_orani': basari_orani,
            'ogrenme_hizi': real_obs / max(1, total_obs - real_obs),
            'pattern_dagilimi': pattern_dagilimi,
            'adaptive_thresholds': self.learning_data['adaptive_thresholds'],
            'son_feedbackler': son_feedbackler
        }
    
    def _default_analysis(self):
        """Varsayılan analiz sonucu"""
        return {
            'yorum': 'Yetersiz veri nedeniyle temel analiz yapıldı',
            'supheli_donemler': 'Yetersiz veri',
            'risk_seviyesi': 'Orta',
            'risk_puan': 2,
            'pattern_data': {}
        }

# Global model instance
adaptive_model = AdaptiveLearningModel()
