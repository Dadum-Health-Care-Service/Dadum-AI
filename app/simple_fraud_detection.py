"""
간단한 이상거래 탐지 모델
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class SimpleFraudDetection:
    """간단한 이상거래 탐지 모델"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "models/simple_fraud_model.pkl"
        self.scaler_path = "models/simple_fraud_scaler.pkl"
        
    def train_model(self, data=None):
        """모델 훈련"""
        try:
            logger.info("모델 훈련 시작...")
            
            # 샘플 데이터 생성 (3000개)
            if data is None:
                data = self.generate_sample_data(3000)
            
            # 특성 추출
            features = []
            for item in data:
                feature = [
                    item['amount'],
                    item['hour'],
                    item['day_of_week'],
                    item['transaction_count_24h'],
                    item['avg_amount_7d'],
                    item['location_distance'],
                    item['card_age_days'],
                    item['merchant_category']
                ]
                features.append(feature)
            
            features = np.array(features)
            
            # 데이터 정규화
            features_scaled = self.scaler.fit_transform(features)
            
            # Isolation Forest 모델 훈련
            self.model = IsolationForest(
                contamination=0.2,  # 20% 이상치 (더 민감하게)
                random_state=42,
                n_estimators=100
            )
            self.model.fit(features_scaled)
            
            # 모델 저장
            self.save_model()
            
            self.is_trained = True
            logger.info("모델 훈련 완료!")
            return True
            
        except Exception as e:
            logger.error(f"모델 훈련 실패: {str(e)}")
            return False
    
    def predict_fraud(self, transaction_data):
        """이상거래 예측"""
        try:
            if not self.is_trained or self.model is None:
                return {
                    "risk_score": 0.0,
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "anomaly_score": 0.0,
                    "recommendation": "모델이 훈련되지 않았습니다. 먼저 모델을 훈련해주세요.",
                    "error": "모델이 훈련되지 않았습니다."
                }
            
            # 특성 추출
            feature = np.array([[
                transaction_data['amount'],
                transaction_data['hour'],
                transaction_data['day_of_week'],
                transaction_data['transaction_count_24h'],
                transaction_data['avg_amount_7d'],
                transaction_data['location_distance'],
                transaction_data['card_age_days'],
                transaction_data['merchant_category']
            ]])
            
            # 데이터 정규화
            feature_scaled = self.scaler.transform(feature)
            
            # 예측
            anomaly_score = self.model.decision_function(feature_scaled)[0]
            is_anomaly = self.model.predict(feature_scaled)[0] == -1
            
            # 위험도 점수 (0-100) - 조정된 공식
            # 더 관대한 위험도 평가
            if is_anomaly:
                # 이상거래: anomaly_score가 음수일 때 위험도 증가
                # -0.5 ~ -0.1 범위를 50-100점으로 매핑 (더 관대하게)
                risk_score = max(50, min(100, 50 + abs(anomaly_score) * 100))
            else:
                # 정상거래: anomaly_score가 양수일 때 위험도 감소
                # 0.1 ~ 0.5 범위를 0-30점으로 매핑 (더 안전하게)
                risk_score = max(0, min(30, anomaly_score * 60))
            
            # 신뢰도
            confidence = abs(anomaly_score) * 100
            
            # 추천사항
            if is_anomaly:
                if risk_score > 80:
                    recommendation = "높은 위험도 이상거래로 판단됩니다. 즉시 거래를 중단하세요."
                elif risk_score > 60:
                    recommendation = "중간 위험도 이상거래로 판단됩니다. 추가 검토가 필요합니다."
                else:
                    recommendation = "낮은 위험도 이상거래로 판단됩니다. 주의 깊게 모니터링하세요."
            else:
                recommendation = "정상 거래로 판단됩니다."
            
            return {
                "risk_score": round(risk_score, 2),
                "is_anomaly": bool(is_anomaly),
                "confidence": round(confidence, 2),
                "anomaly_score": round(anomaly_score, 4),
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            return {
                "risk_score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_score": 0.0,
                "recommendation": "예측 중 오류가 발생했습니다.",
                "error": f"예측 실패: {str(e)}"
            }
    
    def generate_sample_data(self, n_samples):
        """샘플 데이터 생성"""
        data = []
        
        for _ in range(n_samples):
            # 정상 거래 (80%)
            if random.random() < 0.8:
                data.append({
                    'amount': random.randint(1000, 50000),
                    'hour': random.randint(9, 18),  # 업무시간
                    'day_of_week': random.randint(0, 4),  # 평일
                    'transaction_count_24h': random.randint(1, 5),
                    'avg_amount_7d': random.randint(1000, 30000),
                    'location_distance': random.randint(0, 50),
                    'card_age_days': random.randint(365, 3650),
                    'merchant_category': random.randint(1, 10)
                })
            # 이상거래 (20%)
            else:
                data.append({
                    'amount': random.randint(100000, 1000000),  # 큰 금액
                    'hour': random.randint(0, 6),  # 새벽시간
                    'day_of_week': random.randint(5, 6),  # 주말
                    'transaction_count_24h': random.randint(20, 100),  # 많은 거래
                    'avg_amount_7d': random.randint(1000, 10000),
                    'location_distance': random.randint(100, 1000),  # 먼 거리
                    'card_age_days': random.randint(1, 30),  # 새로운 카드
                    'merchant_category': random.randint(15, 20)
                })
        
        return data
    
    def save_model(self):
        """모델 저장"""
        try:
            os.makedirs("models", exist_ok=True)
            
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                logger.info(f"모델 저장: {self.model_path}")
            
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"스케일러 저장: {self.scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"모델 저장 실패: {str(e)}")
            return False
    
    def load_model(self):
        """모델 로드"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info("모델 로드 완료")
                return True
            return False
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            return False

# 전역 인스턴스
fraud_detector = SimpleFraudDetection()
