"""
간단한 AI 서버 메인 애플리케이션
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.simple_api import router as fraud_router
from app.simple_fraud_detection import fraud_detector
from app.ai_yolo import router as ai_router
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simple AI Fraud Detection",
    description="간단한 이상거래 탐지 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:8080",   # Spring Boot 개발 서버
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    logger.info("AI 서버 시작 중...")
    success = fraud_detector.load_model()
    if success:
        logger.info("✅ 모델 로드 완료 - 훈련됨")
    else:
        logger.info("⚠️ 모델 로드 실패 - 미훈련 상태")

# 라우터 등록
app.include_router(fraud_router, prefix="/ai", tags=["fraud-detection"])
app.include_router(ai_router)

@app.get("/")
async def root():
    return {
        "message": "Simple AI Fraud Detection Service",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health/")
async def health():
    return {
        "status": "healthy",
        "service": "Simple AI Fraud Detection",
        "timestamp": "2025-10-14T15:00:00"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
