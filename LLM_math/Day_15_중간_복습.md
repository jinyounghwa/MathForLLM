# Day 15: 중간 복습 - 벡터와 선형대수 (1시간)

## 📚 학습 목표
- Day 11-14의 핵심 개념 총정리
- 개념 간 연결고리 파악하기
- 취약한 부분 찾아 보완하기
- LLM과의 연결성 확인하기

---

## 🎯 강의 주제
**"지금까지 배운 것들의 연결"**

---

## 📖 핵심 개념 정리

### 1. 벡터의 길이 (Day 11)

**노름 (Norm)**:
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**거리**:
```
d(a, b) = ||a - b||
```

**핵심**:
- L2 노름: 유클리드 거리
- 임베딩 공간에서 의미적 거리 측정

---

### 2. 내적 (Day 12)

**정의**:
```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**기하학적 의미**:
```
a · b = ||a|| × ||b|| × cos(θ)
```

**핵심**:
- 내적 크다 = 유사하다
- 내적 = 0 → 직교
- **Attention의 핵심 연산**: QK^T

---

### 3. 정규화 (Day 13)

**단위 벡터**:
```
v̂ = v / ||v||
```

**코사인 유사도**:
```
cos(θ) = (a · b) / (||a|| × ||b||)
```

**핵심**:
- 방향만 보고 싶을 때 정규화
- RAG 시스템의 핵심
- 임베딩 검색에서 필수

---

### 4. 행렬과 행렬곱셈 (Day 14)

**행렬곱**:
```
(m × n) × (n × p) = (m × p)
```

**신경망**:
```
y = Wx + b
```

**핵심**:
- 선형 변환
- 내적의 연속
- 모든 신경망 층의 기초

---

## 🔗 개념 연결 맵

```
벡터의 길이 (||v||)
    ↓
내적 (a · b)
    ↓
각도 (cos(θ) = a·b / ||a||||b||)
    ↓
정규화 (v̂ = v / ||v||)
    ↓
코사인 유사도 (â · b̂)
    ↓
행렬곱 (여러 내적을 한 번에)
    ↓
신경망 (y = Wx + b)
```

**모든 것이 연결되어 있습니다!**

---

## 💻 종합 실습

### 실습 1: 전체 흐름 확인
```python
import numpy as np

print("=== 벡터와 선형대수 종합 실습 ===\n")

# 1. 벡터 정의
a = np.array([3, 4])
b = np.array([1, 2])

print("1. 벡터:")
print(f"   a = {a}, b = {b}\n")

# 2. 길이 (노름)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print("2. 길이 (노름):")
print(f"   ||a|| = {norm_a:.4f}")
print(f"   ||b|| = {norm_b:.4f}\n")

# 3. 거리
distance = np.linalg.norm(a - b)
print("3. 거리:")
print(f"   d(a, b) = ||a - b|| = {distance:.4f}\n")

# 4. 내적
dot_product = np.dot(a, b)
print("4. 내적:")
print(f"   a · b = {dot_product}\n")

# 5. 각도
cos_theta = dot_product / (norm_a * norm_b)
theta_deg = np.degrees(np.arccos(cos_theta))
print("5. 각도:")
print(f"   cos(θ) = {cos_theta:.4f}")
print(f"   θ = {theta_deg:.2f}°\n")

# 6. 정규화
a_normalized = a / norm_a
b_normalized = b / norm_b
print("6. 정규화:")
print(f"   â = {a_normalized}")
print(f"   b̂ = {b_normalized}\n")

# 7. 코사인 유사도
cosine_sim = np.dot(a_normalized, b_normalized)
print("7. 코사인 유사도:")
print(f"   â · b̂ = {cosine_sim:.4f}")
print(f"   (= cos(θ)와 같음)\n")

# 8. 행렬곱
W = np.array([[1, 2],
              [3, 4]])
y = W @ a
print("8. 행렬 변환:")
print(f"   W =\n{W}")
print(f"   y = Wa = {y}\n")

print("✅ 모든 개념이 연결되어 있습니다!")
```

### 실습 2: LLM 시뮬레이션
```python
import numpy as np

print("=== 간단한 Attention 메커니즘 ===\n")

# 임베딩 차원
d = 4

# Query, Key, Value 벡터
Q = np.array([1.0, 0.5, 0.2, 0.1])
K1 = np.array([0.9, 0.6, 0.3, 0.1])
K2 = np.array([0.1, 0.2, 0.8, 0.9])
K3 = np.array([1.0, 0.4, 0.1, 0.0])

V1 = np.array([1, 0, 0, 0])
V2 = np.array([0, 1, 0, 0])
V3 = np.array([0, 0, 1, 0])

print("1. 내적으로 유사도 계산:")
score1 = np.dot(Q, K1)
score2 = np.dot(Q, K2)
score3 = np.dot(Q, K3)
print(f"   Q · K1 = {score1:.4f}")
print(f"   Q · K2 = {score2:.4f}")
print(f"   Q · K3 = {score3:.4f}\n")

# Scaling (Day 43에서 자세히)
d_k = len(Q)
score1 /= np.sqrt(d_k)
score2 /= np.sqrt(d_k)
score3 /= np.sqrt(d_k)

print("2. Scaling (√d_k로 나누기):")
print(f"   scaled_score1 = {score1:.4f}")
print(f"   scaled_score2 = {score2:.4f}")
print(f"   scaled_score3 = {score3:.4f}\n")

# Softmax
scores = np.array([score1, score2, score3])
exp_scores = np.exp(scores)
attention_weights = exp_scores / np.sum(exp_scores)

print("3. Softmax로 가중치 계산:")
print(f"   attention_weights = {attention_weights}\n")

# 가중합
output = attention_weights[0] * V1 + \
         attention_weights[1] * V2 + \
         attention_weights[2] * V3

print("4. Value들의 가중합:")
print(f"   output = {output}\n")

print("✅ 이것이 Attention의 기본 흐름입니다!")
```

---

## ✍️ 자가 진단 문제

### 문제 1: 길이와 거리
```
v = [3, 4]
w = [0, 4]

(a) ||v|| = ?
(b) ||w|| = ?
(c) d(v, w) = ?
```

<details>
<summary>정답</summary>

```
(a) ||v|| = √(9 + 16) = 5
(b) ||w|| = √(0 + 16) = 4
(c) d(v, w) = ||v - w|| = ||[3, 0]|| = 3
```
</details>

### 문제 2: 내적과 각도
```
a = [1, 0]
b = [1, 1]

(a) a · b = ?
(b) 각도 θ = ?
```

<details>
<summary>정답</summary>

```
(a) a · b = 1
(b) cos(θ) = 1 / (1 × √2) = √2/2
    θ = 45°
```
</details>

### 문제 3: 정규화
```
v = [3, 4]

정규화된 v̂ = ?
```

<details>
<summary>정답</summary>

```
||v|| = 5
v̂ = [3/5, 4/5] = [0.6, 0.8]
```
</details>

### 문제 4: 행렬곱
```
A = [1  2]    x = [1]
    [3  4]        [2]

Ax = ?
```

<details>
<summary>정답</summary>

```
Ax = [1×1 + 2×2]  = [5]
     [3×1 + 4×2]    [11]
```
</details>

---

## 🔗 LLM 연결 총정리

### 1. 임베딩 공간
```
- 길이: 단어의 중요도
- 거리: 의미적 유사도
- 방향: 의미 (정규화 후)
```

### 2. Attention 메커니즘
```
1. 내적: Q · K (유사도)
2. Scaling: / √d_k
3. Softmax: 가중치
4. 가중합: Σ (weight × V)
```

### 3. 신경망 층
```
y = Wx + b
- 입력 변환
- 차원 변경
- 특징 추출
```

---

## ✅ 체크포인트

- [ ] **벡터의 길이를 계산할 수 있나요?**

- [ ] **내적의 기하학적 의미를 설명할 수 있나요?**

- [ ] **코사인 유사도를 계산할 수 있나요?**

- [ ] **행렬곱의 차원을 맞출 수 있나요?**

- [ ] **이 모든 개념이 Attention에서 어떻게 쓰이는지 이해했나요?**

---

## 🎓 핵심 요약

**지금까지 배운 것**:
1. 벡터의 길이와 거리
2. 내적과 유사도
3. 정규화와 코사인 유사도
4. 행렬곱과 선형 변환

**이 모든 것이 LLM의 기초입니다!**

### 다음 학습
- **Day 16-20**: 행렬의 다양한 성질
  - 전치, 역행렬, 고유값/고유벡터

---

**수고하셨습니다!** 🎉

**복습을 통해 더 깊이 이해했습니다!**
