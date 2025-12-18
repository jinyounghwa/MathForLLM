# Day 14: 행렬과 행렬곱셈 (1.5시간)

## 📚 학습 목표
- 행렬의 정의와 표현 이해하기
- 행렬곱셈의 규칙과 의미 파악하기
- 신경망에서 y = Wx + b의 의미 이해하기
- 행렬 차원의 중요성 깨닫기

---

## 🎯 강의 주제
**"벡터 여러 개를 한 번에 다루기"**

---

## 📖 핵심 개념

### 1. 행렬이란?

#### 1.1 정의
**숫자들을 직사각형 배열로 나열한 것**

```
A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]

m × n 행렬: m개의 행(row), n개의 열(column)
```

**예시**:
```
A = [1  2  3]  ← 2×3 행렬
    [4  5  6]

B = [1  2]     ← 3×2 행렬
    [3  4]
    [5  6]
```

#### 1.2 벡터 관점
**행렬 = 여러 벡터의 모음**

**열 벡터들의 모음**:
```
A = [v₁  v₂  v₃]

v₁ = [1]  v₂ = [2]  v₃ = [3]
     [4]       [5]       [6]
```

**행 벡터들의 모음**:
```
A = [—— u₁ ——]
    [—— u₂ ——]

u₁ = [1  2  3]
u₂ = [4  5  6]
```

---

### 2. 행렬곱셈 ⭐

#### 2.1 규칙
**행렬 A (m×n)과 B (n×p)의 곱 = C (m×p)**

```
(m × n) × (n × p) = (m × p)
        ↑
    일치해야 함!
```

**계산 방법**:
```
C[i,j] = A의 i번째 행 · B의 j번째 열
```

**예시**:
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

C = AB = [(1×5+2×7)  (1×6+2×8)]  = [19  22]
         [(3×5+4×7)  (3×6+4×8)]    [43  50]
```

**단계별 계산**:
```
C[0,0] = [1  2] · [5] = 1×5 + 2×7 = 19
                  [7]

C[0,1] = [1  2] · [6] = 1×6 + 2×8 = 22
                  [8]

C[1,0] = [3  4] · [5] = 3×5 + 4×7 = 43
                  [7]

C[1,1] = [3  4] · [6] = 3×6 + 4×8 = 50
                  [8]
```

#### 2.2 행렬곱셈의 의미

**1. 선형 변환**:
```
y = Ax

x를 A로 변환하여 y를 얻음
```

**2. 벡터들의 가중 합**:
```
A = [a₁  a₂  a₃]
x = [x₁]
    [x₂]
    [x₃]

Ax = x₁a₁ + x₂a₂ + x₃a₃
```

**3. 내적의 연속**:
```
y[i] = A의 i번째 행 · x
```

---

### 3. 행렬곱셈의 성질

**1. 결합법칙**:
```
(AB)C = A(BC)
```

**2. 분배법칙**:
```
A(B + C) = AB + AC
(A + B)C = AC + BC
```

**3. 교환법칙은 성립하지 않음!**:
```
AB ≠ BA (일반적으로)
```

**예시**:
```
A = [1  2]    B = [0  1]
    [0  0]        [0  0]

AB = [0  1]    BA = [0  0]
     [0  0]         [0  0]

AB ≠ BA
```

---

### 4. 신경망의 선형 층

#### 4.1 기본 형태
```
y = Wx + b

W: 가중치 행렬 (weight)
x: 입력 벡터
b: 편향 벡터 (bias)
y: 출력 벡터
```

**예시**:
```
입력 차원: 3
출력 차원: 2

W = [w₁₁  w₁₂  w₁₃]  (2×3)
    [w₂₁  w₂₂  w₂₃]

x = [x₁]  (3×1)
    [x₂]
    [x₃]

b = [b₁]  (2×1)
    [b₂]

y = Wx + b = [w₁₁x₁ + w₁₂x₂ + w₁₃x₃ + b₁]  (2×1)
             [w₂₁x₁ + w₂₂x₂ + w₂₃x₃ + b₂]
```

#### 4.2 배치 처리
```
Y = XW^T + b

X: 배치 행렬 (batch_size × input_dim)
W^T: 전치된 가중치 (input_dim × output_dim)
b: 편향 (output_dim) - 브로드캐스팅
Y: 출력 (batch_size × output_dim)
```

---

## 💻 Python 실습

### 실습 1: 행렬곱셈 기초
```python
import numpy as np

# 행렬 정의
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("=== 행렬곱셈 ===")
print(f"A =\n{A}\n")
print(f"B =\n{B}\n")

# 행렬곱
C = A @ B  # 또는 np.dot(A, B)
print(f"C = A @ B =\n{C}\n")

# 수동 계산
C_manual = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        C_manual[i, j] = np.dot(A[i, :], B[:, j])

print(f"수동 계산:\n{C_manual}\n")
print(f"결과 일치: {np.allclose(C, C_manual)}")
```

### 실습 2: 신경망 선형 층 시뮬레이션
```python
import numpy as np

# 선형 층: 3차원 → 2차원
input_dim = 3
output_dim = 2

# 가중치와 편향 초기화
W = np.random.randn(output_dim, input_dim) * 0.1
b = np.zeros(output_dim)

# 입력 벡터
x = np.array([1.0, 2.0, 3.0])

print("=== 신경망 선형 층 ===")
print(f"입력 차원: {input_dim}")
print(f"출력 차원: {output_dim}\n")
print(f"W (가중치) =\n{W}\n")
print(f"b (편향) = {b}\n")
print(f"x (입력) = {x}\n")

# 순전파
y = W @ x + b

print(f"y = Wx + b = {y}\n")
print("각 출력 계산:")
for i in range(output_dim):
    print(f"  y[{i}] = {W[i,:]} · {x} + {b[i]:.2f}")
    print(f"       = {np.dot(W[i,:], x):.4f} + {b[i]:.2f}")
    print(f"       = {y[i]:.4f}")
```

### 실습 3: 배치 처리
```python
import numpy as np

# 배치 처리: 여러 입력을 한 번에
batch_size = 4
input_dim = 3
output_dim = 2

# 가중치
W = np.random.randn(output_dim, input_dim) * 0.1
b = np.random.randn(output_dim) * 0.1

# 배치 입력 (4개 샘플)
X = np.random.randn(batch_size, input_dim)

print("=== 배치 처리 ===")
print(f"배치 크기: {batch_size}")
print(f"입력 차원: {input_dim}")
print(f"출력 차원: {output_dim}\n")

print(f"X 형태: {X.shape}")
print(f"W 형태: {W.shape}")
print(f"b 형태: {b.shape}\n")

# 배치 순전파
Y = X @ W.T + b  # (4,3) @ (3,2) + (2,) = (4,2)

print(f"Y = XW^T + b")
print(f"Y 형태: {Y.shape}\n")
print(f"Y =\n{Y}\n")

print("각 샘플별 출력:")
for i in range(batch_size):
    y_i = W @ X[i] + b
    print(f"  샘플 {i}: {y_i}")
```

---

## ✍️ 손 계산 연습

### 연습 1: 행렬곱셈
```
A = [1  2]    B = [2]
    [3  4]        [1]

AB = [1×2 + 2×1]  = [4]
     [3×2 + 4×1]    [10]
```

### 연습 2: 차원 확인
```
A: (2×3)  B: (3×4)  → AB: (2×4) ✓

A: (2×3)  C: (2×4)  → AC: 불가능! (3 ≠ 2) ✗
```

### 연습 3: 신경망 계산
```
W = [1  2  3]    x = [1]    b = [0]
    [4  5  6]        [2]        [1]
                     [3]

y = Wx + b = [(1×1 + 2×2 + 3×3) + 0]  = [14]
             [(4×1 + 5×2 + 6×3) + 1]    [33]
```

---

## 🔗 LLM 연결점

### 1. 신경망의 모든 층
```
# 선형 층
y = Wx + b

# Attention에서
Q = XW_Q
K = XW_K
V = XW_V

모두 행렬곱!
```

### 2. 임베딩 테이블
```
Embedding(vocab_size, embed_dim)

E: (vocab_size × embed_dim)
idx: 토큰 인덱스

embedding_vector = E[idx]  (lookup)
```

### 3. 배치 처리의 효율성
```
# 하나씩 처리 (느림)
for x in batch:
    y = W @ x + b

# 배치로 처리 (빠름)
Y = X @ W.T + b  # 병렬 계산!
```

---

## ✅ 체크포인트

- [ ] **행렬곱셈의 차원 규칙을 이해했나요?**

- [ ] **행렬곱셈을 손으로 계산할 수 있나요?**

- [ ] **y = Wx + b의 의미를 설명할 수 있나요?**

- [ ] **배치 처리의 장점을 이해했나요?**

---

## 🎓 핵심 요약

1. **행렬**: 벡터들의 모음
2. **행렬곱**: (m×n) × (n×p) = (m×p)
3. **계산**: C[i,j] = A의 i행 · B의 j열
4. **신경망**: y = Wx + b
5. **배치**: 여러 입력을 한 번에 처리

### 다음 학습
- **Day 15**: 중간 복습
  - 벡터와 선형대수 총정리

---

**수고하셨습니다!** 🎉

**행렬곱셈은 신경망의 기초입니다!**
