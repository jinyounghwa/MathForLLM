# Day 29: 베이즈 정리 (1.5시간)

## 📚 학습 목표
- 베이즈 정리 이해하기
- 사전/사후 확률 개념 파악하기
- LLM의 토큰 예측에 적용하기

---

## 🎯 강의 주제
**"관찰로부터 믿음 업데이트하기"**

---

## 📖 핵심 개념

### 베이즈 정리
```
P(A|B) = P(B|A) × P(A) / P(B)

P(A): 사전 확률 (prior)
P(B|A): 가능도 (likelihood)
P(A|B): 사후 확률 (posterior)
P(B): 증거 (evidence)
```

### LLM에서의 베이즈
```
P(token | context) ∝ P(context | token) × P(token)

Transformer는 이를 직접 모델링!
```

---

## 💻 Python 실습

```python
import numpy as np

# 예: 스팸 필터
# P(스팸|"무료") = ?

p_spam = 0.3  # 사전 확률
p_not_spam = 0.7

p_free_given_spam = 0.8  # 가능도
p_free_given_not_spam = 0.1

# 전체 확률
p_free = p_free_given_spam * p_spam + p_free_given_not_spam * p_not_spam

# 베이즈 정리
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

print("=== 베이즈 정리: 스팸 필터 ===\n")
print(f"P(스팸) = {p_spam}")
print(f"P('무료'|스팸) = {p_free_given_spam}")
print(f"P(스팸|'무료') = {p_spam_given_free:.4f}")
```

---

## 🔗 LLM 연결점

### 다음 토큰 예측
```python
# 간소화된 예
context = "I love"
tokens = ["you", "pizza", "math"]

# P(token | context)를 베이즈로 해석 가능
```

---

## 🎓 핵심 요약

**베이즈 정리**: 관찰로 믿음을 업데이트
- 사전 → 관찰 → 사후

### 다음 학습
- **Day 30**: 확률분포

---

**베이즈는 LLM 예측의 철학적 기초입니다!**
