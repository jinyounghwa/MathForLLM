# MathForLLM - LLM을 위한 수학 기초 학습 웹서비스

![Status](https://img.shields.io/badge/Status-Development-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

LLM/AI 개발을 위한 수학 기초를 효과적으로 학습할 수 있는 대화형 웹 기반 학습 서비스입니다.

## 🎯 핵심 기능

### 1. 일반 질문 모드
- 자유로운 형식의 수학 질문 가능
- RAG 기반 정확한 답변 제공
- 참고 자료 자동 인용
- 질문 히스토리 저장

### 2. 계획된 학습 모드
- 체계적인 커리큘럼 기반 학습
- 맞춤형 학습 일정 자동 생성
- 학습 진도 추적 및 시각화
- 일차별 학습 내용 자동 할당

### 3. 답변 모드
- **일반 모드**: 직관적이고 간결한 설명
- **역할극 모드**: 실제 상황을 재현한 몰입형 학습

## 🛠️ 기술 스택

### 프론트엔드
- **Framework**: Next.js 16
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **Storage**: LocalStorage

### 백엔드
- **Framework**: Hono (Express보다 경량)
- **Runtime**: Node.js
- **Vector DB**: Vectra (경량, 로컬 파일 기반)
- **LLM**: Ollama (Qwen 2.5 7B)
- **Language**: TypeScript

### 학습 자료
- 44개의 구조화된 마크다운 파일
- Day 1 ~ Day 46+ 커리큘럼
- 기초 수학부터 트랜스포머 아키텍처까지

## 📋 시스템 구조

```
┌─────────────────────────┐
│   Frontend (Next.js)    │
│  - 채팅 인터페이스      │
│  - 학습 모드 선택       │
│  - 진도 추적            │
└────────────┬────────────┘
             │ HTTP
┌────────────▼────────────┐
│  Backend (Hono)         │
│  - /api/chat            │
│  - /api/curriculum      │
│  - RAG 엔진             │
└────────────┬────────────┘
      ┌──────┴──────┐
      │             │
┌─────▼────┐  ┌─────▼──────┐
│  Vectra   │  │  Ollama    │
│ (Vector   │  │ (Qwen 2.5) │
│   DB)     │  │            │
└──────────┘  └────────────┘
```

## 🚀 설치 및 실행

### 1. 선행 요구사항
- Node.js 18+
- Ollama (로컬 LLM 실행)

### 2. Ollama 설치 및 실행

```bash
# Ollama 다운로드: https://ollama.ai

# Qwen 2.5 7B 모델 다운로드 및 실행
ollama run qwen2.5:7b
```

### 3. 프로젝트 설정

```bash
# 백엔드 설정
cd backend
npm install
npm run ingest  # 문서 인제스션 (RAG 벡터 인덱싱)
npm run dev     # 백엔드 서버 시작 (포트 3001)

# 새로운 터미널에서 프론트엔드 설정
cd frontend
npm install
npm run dev     # 프론트엔드 서버 시작 (포트 3000)
```

### 4. 접속
- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:3001

## 📖 사용 방법

### 일반 질문 모드
1. 홈 페이지에서 "일반 질문 모드" 선택
2. 모드 선택 (일반 / 역할극)
3. 수학 질문 입력
4. AI 멘토의 답변 받기

### 계획된 학습 모드
1. 홈 페이지에서 "계획된 학습 모드" 선택
2. 학습 설정:
   - 학습 주기: 매일 / 2일마다 / 3일마다 / 주 1회
   - 학습 시간: 30분 / 1시간 / 2시간
   - 시작일 선택
3. 학습 계획 자동 생성
4. 일차별로 학습 진행

## 🔧 백엔드 API

### POST `/api/chat`
사용자 질문에 대한 답변 생성

**Request:**
```json
{
  "message": "엔트로피란 무엇인가요?",
  "mode": "roleplay",
  "learningMode": "free",
  "history": []
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [
    {
      "file": "Day_35_정보_이득.md",
      "section": "엔트로피",
      "relevance": 0.95
    }
  ],
  "suggestedQuestions": ["...", "..."]
}
```

### POST `/api/curriculum`
학습 계획 생성

**Request:**
```json
{
  "frequency": 2,
  "duration": 60,
  "startDate": "2024-12-19"
}
```

**Response:**
```json
{
  "curriculumId": "uuid",
  "totalDays": 90,
  "schedule": [
    {
      "day": 1,
      "date": "2024-12-19",
      "topic": "수의 체계와 기초 연산",
      "sections": ["수의 체계와 기초 연산"],
      "estimatedTime": 60,
      "completed": false
    }
  ]
}
```

## 📁 프로젝트 구조

```
math-for-llm/
├── frontend/                 # Next.js 프론트엔드
│   ├── app/
│   │   ├── page.tsx         # 홈 페이지
│   │   ├── chat/page.tsx    # 일반 질문 모드
│   │   └── curriculum/      # 계획된 학습 모드
│   ├── components/          # 재사용 가능 컴포넌트
│   ├── lib/                 # 유틸리티 함수
│   └── package.json
│
├── backend/                  # Hono 백엔드
│   ├── src/
│   │   ├── index.ts         # 메인 서버
│   │   ├── routes/          # API 라우트
│   │   ├── services/        # 비즈니스 로직
│   │   │   ├── rag.ts       # RAG 엔진
│   │   │   ├── llm.ts       # LLM 통합
│   │   │   └── vectordb.ts  # 벡터 DB
│   │   └── scripts/         # 유틸리티 스크립트
│   └── package.json
│
├── LLM_math/                # RAG 교재 파일 (44개 MD)
│   ├── Day_01_수의_체계.md
│   ├── Day_02-03_지수와_로그.md
│   └── ...
│
└── README.md
```

## 🧠 학습 커리큘럼 개요

1. **기초 수학** (Day 1-6): 수의 체계, 지수와 로그
2. **선형대수** (Day 7-20): 벡터, 행렬, 고유값
3. **미적분** (Day 21-25): 극한, 미분, 최적화
4. **확률 통계** (Day 26-32): 확률, 분포, 베이즈 정리
5. **신경망** (Day 33-46+): 활성화 함수, 백프로퍼게이션, 트랜스포머

## 🔐 보안 및 프라이버시

- 로컬 LLM 사용 (클라우드 송신 없음)
- 질문 히스토리 로컬 저장
- 개인정보 수집 없음
- 오픈소스 기반 구축

## 📊 성능 지표

- 답변 생성 시간: < 3초
- 메모리 사용: < 2GB
- RAG 검색 정확도: 80%+
- 모바일 친화적 UI

## 🐛 알려진 이슈 및 해결 방법

### Ollama 연결 실패
```bash
# 1. Ollama가 실행 중인지 확인
ollama list

# 2. 모델이 설치되어 있는지 확인
ollama run qwen2.5:7b

# 3. OLLAMA_URL 환경 변수 확인
echo $OLLAMA_URL
```

### 벡터 인덱싱 실패
```bash
# 1. 백엔드에서 문서 인제스션 재시도
cd backend
npm run ingest

# 2. 인덱스 파일 확인
ls -la /tmp/vectra_db/
```

## 📝 환경 변수

### 백엔드 (.env)
```
OLLAMA_URL=http://localhost:11434
PORT=3001
```

### 프론트엔드 (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:3001
```

## 📄 라이선스

MIT License - 자유로운 사용 가능

## 👨‍💻 개발자

**진영화 (Younghwa Jin)**
- AI/ML 엔지니어
- LLM 기반 학습 플랫폼 개발

---

**마지막 업데이트**: 2024-12-18
**버전**: 1.0.0-beta
