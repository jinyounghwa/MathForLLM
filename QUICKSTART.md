# 🚀 빠른 시작 가이드

MathForLLM을 3분 안에 실행하세요!

## 필수 요구사항

1. **Node.js 18+** 설치: https://nodejs.org
2. **Ollama 설치**: https://ollama.ai
3. **Qwen 2.5 7B 모델** (이미 설치됨)

## Step 1: Ollama 실행

```bash
# Ollama 실행 (새 터미널)
ollama run qwen2.5:7b
```

Ollama가 시작될 때까지 잠시 기다리세요.

## Step 2: 백엔드 시작

```bash
# 새 터미널에서
cd backend

# 의존성 설치
npm install

# 문서 인제스션 (RAG 벡터 인덱싱) - 첫 실행 시에만 필요
npm run ingest

# 백엔드 서버 시작
npm run dev
```

성공 메시지: `🚀 Server is running on http://localhost:3001`

## Step 3: 프론트엔드 시작

```bash
# 새 터미널에서
cd frontend

# 의존성 설치
npm install

# 프론트엔드 서버 시작
npm run dev
```

성공 메시지: `▲ Next.js 16.0.10`

## Step 4: 브라우저에서 접속

http://localhost:3000 을 열고 학습을 시작하세요! 🎓

---

## 문제 해결

### 포트 이미 사용 중
```bash
# 포트 변경
npm run dev -- -p 3002
```

### Ollama 연결 안됨
```bash
# Ollama 상태 확인
ollama list

# 모델 재설치
ollama run qwen2.5:7b
```

### 벡터 인덱싱 오류
```bash
# 백엔드에서 재시도
cd backend
npm run ingest
```

---

## 기능 테스트

### 일반 질문 모드 테스트
1. 홈페이지 → "일반 질문 모드" 선택
2. "엔트로피란 무엇인가요?"라고 질문
3. AI 멘토의 답변 확인

### 계획된 학습 모드 테스트
1. 홈페이지 → "계획된 학습 모드" 선택
2. 설정: 2일마다, 1시간, 오늘 시작
3. "학습 계획 생성하기" 클릭
4. 오늘의 학습 시작

---

## 프로덕션 배포

### 프론트엔드 (Vercel)
```bash
cd frontend
npm run build
vercel deploy
```

### 백엔드 (로컬 서버)
```bash
cd backend
npm run build
npm start
```

---

**모든 준비가 완료되었습니다! 🎉**

문제가 있으면 README.md의 "알려진 이슈" 섹션을 참고하세요.
