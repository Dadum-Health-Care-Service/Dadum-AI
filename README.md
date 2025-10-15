# Dadum AI Server

AI 기반 이상거래 탐지 및 이상 행동 탐지 서비스

## 🚀 기능

- **이상거래 탐지**: 실시간 거래 데이터 분석을 통한 이상거래 탐지
- **배치 처리**: 다수의 거래를 한 번에 분석
- **머신러닝 모델**: Isolation Forest 알고리즘 기반 이상치 탐지
- **앙상블 모델**: 다중 알고리즘을 통한 고정밀 탐지
- **실시간 학습**: 새로운 데이터로 모델 자동 업데이트
- **고급 분석**: 트렌드 분석, 패턴 탐지, 예측 분석
- **보안 강화**: JWT 인증, 암호화, 감사 로그
- **REST API**: FastAPI 기반의 고성능 API 서버

## 📋 요구사항

- Python 3.8+
- pip

## 🛠️ 설치 및 실행

### 1. 가상환경 생성 및 활성화
```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
# 개발 모드 (자동 재시작)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📡 API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs` (Swagger UI)

### 주요 엔드포인트

#### 1. 이상거래 탐지
```
POST /ai/detect-fraud
```

**요청 예시:**
```json
{
  "transaction_id": "tx_12345",
  "amount": 50000,
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "hour": 10,
  "day_of_week": 1,
  "transaction_count_24h": 3,
  "avg_amount_7d": 25000,
  "location_distance": 5.2,
  "card_age_days": 365,
  "merchant_category": 12
}
```

**응답 예시:**
```json
{
  "transaction_id": "tx_12345",
  "risk_score": 75.5,
  "is_anomaly": true,
  "confidence": 85.2,
  "anomaly_score": -0.3,
  "recommendation": "⚠️ 중간 위험도: 추가 인증을 요청하고 거래를 모니터링하세요.",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### 2. 배치 이상거래 탐지
```
POST /ai/detect-fraud-batch
```

#### 3. 모델 상태 확인
```
GET /ai/model-status
```

#### 4. 모델 훈련
```
POST /ai/train-model
```

#### 5. 서버 상태 확인
```
GET /health/
```


## 🔧 설정

### 환경 변수

프로젝트는 환경변수를 통한 설정을 지원합니다. `env.example` 파일을 참고하여 `.env` 파일을 생성하세요.

```bash
# .env 파일 생성
cp env.example .env

# .env 파일 편집
nano .env  # 또는 원하는 에디터 사용
```

**주요 환경변수:**
```bash
# Gmail 설정
GMAIL_USERNAME=your-email@gmail.com
GMAIL_PASSWORD=your-app-password
GMAIL_SENDER_NAME=Dadum AI System

# 관리자 설정
ADMIN_EMAILS=admin1@example.com,admin2@example.com
ADMIN_PHONE_NUMBERS=01012345678,01098765432

# AI 모델 설정
AI_MODEL_PATH=models/fraud_model.pkl
AI_SCALER_PATH=models/fraud_scaler.pkl
AI_CONTAMINATION=0.1

# 서버 설정
DEBUG=True
HOST=0.0.0.0
PORT=8000
```


### CORS 설정
현재 다음 도메인에서 접근 가능:
- `http://localhost:3000` (React 개발 서버)
- `http://localhost:8080` (Spring Boot 개발 서버)

## 📊 모델 정보

### 사용 알고리즘
- **Isolation Forest**: 이상치 탐지에 특화된 앙상블 알고리즘
- **StandardScaler**: 데이터 정규화

### 특징 (Features)
1. 거래 금액 (amount)
2. 거래 시간 (hour)
3. 요일 (day_of_week)
4. 24시간 내 거래 횟수 (transaction_count_24h)
5. 7일 평균 거래 금액 (avg_amount_7d)
6. 위치 거리 (location_distance)
7. 카드 사용 기간 (card_age_days)
8. 상점 카테고리 (merchant_category)

## 🚨 위험도 기준

- **80% 이상**: 🚨 높은 위험도 - 즉시 차단
- **60-79%**: ⚠️ 중간 위험도 - 추가 인증 필요
- **40-59%**: 🔍 낮은 위험도 - 모니터링 권장
- **40% 미만**: ✅ 정상 거래

## 🔄 Spring Boot 연동

Spring Boot에서 AI 서버를 호출하는 예시:

```java
@Service
public class AIService {
    
    @Autowired
    private RestTemplate restTemplate;
    
    @Value("${ai.service.url:http://localhost:8000}")
    private String aiServiceUrl;
    
    public FraudResult detectFraud(Transaction transaction) {
        String url = aiServiceUrl + "/ai/detect-fraud";
        return restTemplate.postForObject(url, transaction, FraudResult.class);
    }
}
```

## 🔄 React 연동

React에서 AI 서버를 호출하는 예시:

```javascript
const detectFraud = async (transaction) => {
  const response = await fetch('http://localhost:8000/ai/detect-fraud', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction)
  });
  return response.json();
};
```


## 📈 성능 최적화

- **모델 캐싱**: 훈련된 모델을 메모리에 로드하여 빠른 예측
- **배치 처리**: 다수의 거래를 한 번에 처리
- **비동기 처리**: FastAPI의 비동기 처리로 높은 처리량

## 🐛 문제 해결

### 모델이 훈련되지 않은 경우
```bash
# 모델 훈련 API 호출
curl -X POST http://localhost:8000/ai/train-model
```

### 메모리 부족 오류
```bash
# Python 메모리 제한 설정
export PYTHONHASHSEED=0
```

## 📝 로그

서버 실행 시 다음과 같은 로그가 출력됩니다:
- 모델 로드 상태
- API 요청/응답 로그
- 오류 메시지

## 🤝 기여

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License
