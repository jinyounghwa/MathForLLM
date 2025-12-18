# Day 40: WordPiece와 SentencePiece (1.5시간)

## 📚 학습 목표
- WordPiece와 SentencePiece 이해하기
- BPE와의 차이점 파악하기

---

## 🎯 강의 주제
**"BPE의 변형들"**

---

## 📖 핵심 개념

### 1. WordPiece (BERT)
```
BPE와 유사하지만:
- 빈도 대신 가능도(likelihood) 최대화
- 병합 기준: log P(문장) 증가량
```

### 2. SentencePiece (다국어)
```
특징:
- 공백도 토큰으로 처리
- 언어 독립적
- Unigram LM 또는 BPE
```

---

## 💻 Python 실습

```python
# SentencePiece 예시 (설치 필요: pip install sentencepiece)
try:
    import sentencepiece as spm

    # 학습 (실제로는 큰 말뭉치 사용)
    # spm.SentencePieceTrainer.train(
    #     '--input=corpus.txt --model_prefix=m --vocab_size=1000'
    # )

    print("SentencePiece는 공백도 토큰화!")
    print("예: 'Hello world' → ['▁Hello', '▁world']")

except ImportError:
    print("sentencepiece 미설치")
```

---

## 🔗 LLM 연결점

| 모델 | 토크나이저 |
|------|-----------|
| GPT | BPE |
| BERT | WordPiece |
| T5 | SentencePiece (Unigram) |
| LLaMA | SentencePiece (BPE) |

---

## 🎓 핵심 요약

**세 가지 방법 모두 서브워드 토큰화**
- BPE: 빈도 기반
- WordPiece: 가능도 기반
- SentencePiece: 언어 독립적

### 다음 학습
- **Day 41**: 한국어 토크나이저

---

**토크나이저 선택이 성능에 영향을 줍니다!**
