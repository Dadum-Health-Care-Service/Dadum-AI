"""
간단한 이상거래 탐지 API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime

from app.simple_fraud_detection import fraud_detector

logger = logging.getLogger(__name__)
router = APIRouter()

# 요청 모델
class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    user_id: str
    timestamp: str
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    transaction_count_24h: Optional[int] = 1
    avg_amount_7d: Optional[float] = None
    location_distance: Optional[float] = 0
    card_age_days: Optional[int] = 365
    merchant_category: Optional[int] = 1

# 응답 모델
class FraudResponse(BaseModel):
    transaction_id: str
    risk_score: float
    is_anomaly: bool
    confidence: float
    anomaly_score: Optional[float] = None
    recommendation: str
    timestamp: str
    error: Optional[str] = None

@router.post("/detect-fraud", response_model=FraudResponse)
async def detect_fraud(transaction: TransactionRequest):
    """이상거래 탐지"""
    try:
        # 현재 시간 정보 추가
        now = datetime.now()
        hour = transaction.hour if transaction.hour is not None else now.hour
        day_of_week = transaction.day_of_week if transaction.day_of_week is not None else now.weekday()
        
        # 거래 데이터 구성
        transaction_data = {
            'amount': transaction.amount,
            'hour': hour,
            'day_of_week': day_of_week,
            'transaction_count_24h': transaction.transaction_count_24h,
            'avg_amount_7d': transaction.avg_amount_7d or transaction.amount,
            'location_distance': transaction.location_distance,
            'card_age_days': transaction.card_age_days,
            'merchant_category': transaction.merchant_category
        }
        
        # 이상거래 탐지
        result = fraud_detector.predict_fraud(transaction_data)
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            risk_score=result['risk_score'],
            is_anomaly=result['is_anomaly'],
            confidence=result['confidence'],
            anomaly_score=result.get('anomaly_score'),
            recommendation=result['recommendation'],
            timestamp=datetime.now().isoformat(),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"이상거래 탐지 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이상거래 탐지 실패: {str(e)}")

@router.post("/train-model")
async def train_model():
    """모델 훈련"""
    try:
        logger.info("모델 훈련 시작...")
        
        success = fraud_detector.train_model()
        
        if success:
            return {
                "success": True,
                "message": "모델 훈련이 완료되었습니다.",
                "is_trained": fraud_detector.is_trained,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "모델 훈련에 실패했습니다.",
                "is_trained": False,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"모델 훈련 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"모델 훈련 실패: {str(e)}")

@router.get("/model-status")
async def get_model_status():
    """모델 상태 확인"""
    try:
        return {
            "is_trained": fraud_detector.is_trained,
            "model_path": fraud_detector.model_path,
            "scaler_path": fraud_detector.scaler_path,
            "status": "ready" if fraud_detector.is_trained else "not_trained",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"모델 상태 확인 실패: {str(e)}")
        return {
            "is_trained": False,
            "error": f"모델 상태 확인 실패: {str(e)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }

@router.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "Simple Fraud Detection",
        "timestamp": datetime.now().isoformat()
    }
