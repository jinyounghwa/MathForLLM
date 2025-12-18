# Day 18: 고유값과 고유벡터 (Eigenvalues & Eigenvectors) (1시간)

## 📚 학습 목표
- 고유값과 고유벡터의 정의 이해하기
- 계산 방법 익히기
- 기하학적 의미 파악하기
- 주성분 분석(PCA)과의 연결 이해하기

---

## 🎯 강의 주제
**"행렬이 방향만 유지하는 특별한 벡터"**

---

## 📖 핵심 개념

### 1. 고유벡터와 고유값

**정의**:
```
Av = λv

A: 행렬
v: 고유벡터 (eigenvector)
λ: 고유값 (eigenvalue)
```

**의미**:
- 행렬 A를 벡터 v에 곱하면
- 방향은 유지되고
- 크기만 λ배가 됨

**시각화**:
```
v → Av = λv

λ > 1: 늘어남
λ = 1: 그대로
0 < λ < 1: 줄어듦
λ < 0: 반대 방향
```

---

### 2. 고유값 계산

**특성 방정식**:
```
Av = λv
(A - λI)v = 0

v ≠ 0이려면:
det(A - λI) = 0
```

**2×2 예시**:
```
A = [3  1]
    [0  2]

det(A - λI) = det([3-λ    1  ])
                 [0    2-λ]

= (3-λ)(2-λ) - 0
= λ² - 5λ + 6
= (λ-2)(λ-3)

∴ λ₁ = 2, λ₂ = 3
```

---

### 3. 고유벡터 계산

**각 고유값에 대해**:
```
(A - λI)v = 0 을 풀기
```

**λ = 2일 때**:
```
(A - 2I)v = 0

[1  1] [v₁]   [0]
[0  0] [v₂] = [0]

v₁ + v₂ = 0
→ v = [1]  (또는 임의의 스칼라배)
      [-1]
```

**λ = 3일 때**:
```
(A - 3I)v = 0

[0  1] [v₁]   [0]
[0 -1] [v₂] = [0]

v₂ = 0
→ v = [1]  (또는 임의의 스칼라배)
      [0]
```

---

### 4. 기하학적 의미

**대각 행렬**:
```
D = [2  0]
    [0  3]

Dv₁ = [2  0] [1]   [2]   = 2v₁
      [0  3] [0] = [0]

Dv₂ = [2  0] [0]   [0]   = 3v₂
      [0  3] [1] = [3]
```

**대각화 (Diagonalization)**:
```
A = PDP⁻¹

P: 고유벡터들을 열로 가진 행렬
D: 고유값들을 대각선에 가진 행렬
```

---

## 💻 Python 실습

### 실습 1: 고유값과 고유벡터 계산
```python
import numpy as np

A = np.array([[3, 1],
              [0, 2]])

print("=== 고유값과 고유벡터 ===")
print(f"A =\n{A}\n")

# NumPy로 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"고유값: {eigenvalues}")
print(f"\n고유벡터:\n{eigenvectors}\n")

# 확인: Av = λv
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]

    Av = A @ v
    λv = λ * v

    print(f"λ_{i+1} = {λ:.4f}")
    print(f"v_{i+1} = {v}")
    print(f"Av  = {Av}")
    print(f"λv  = {λv}")
    print(f"일치: {np.allclose(Av, λv)}\n")
```

### 실습 2: 대각화
```python
import numpy as np

A = np.array([[3, 1],
              [0, 2]])

print("=== 대각화 ===")
print(f"A =\n{A}\n")

# 고유값 분해
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)

print(f"P (고유벡터) =\n{P}\n")
print(f"D (고유값) =\n{D}\n")

# A = PDP⁻¹ 확인
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

print(f"PDP⁻¹ =\n{A_reconstructed}\n")
print(f"A와 일치: {np.allclose(A, A_reconstructed)}")
```

### 실습 3: 공분산 행렬의 고유벡터 (PCA 예고)
```python
import numpy as np
import matplotlib.pyplot as plt

# 상관관계 있는 2D 데이터 생성
np.random.seed(42)
mean = [0, 0]
cov = [[2, 1.5],
       [1.5, 2]]
data = np.random.multivariate_normal(mean, cov, 100)

print("=== 공분산 행렬의 고유벡터 ===")

# 공분산 행렬
C = np.cov(data.T)
print(f"공분산 행렬:\n{C}\n")

# 고유값, 고유벡터
eigenvalues, eigenvectors = np.linalg.eig(C)

print(f"고유값: {eigenvalues}")
print(f"고유벡터:\n{eigenvectors}\n")

# 시각화
plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)

# 고유벡터 그리기 (주성분 방향)
for i in range(2):
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    plt.arrow(0, 0, v[0], v[1], head_width=0.3,
              head_length=0.3, fc=f'C{i+1}', ec=f'C{i+1}',
              linewidth=3, label=f'주성분 {i+1}')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('데이터와 주성분 (고유벡터)')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('pca_eigenvectors.png', dpi=150)
print("시각화 저장 완료!")
```

---

## ✍️ 손 계산 연습

### 연습 1: 고유값
```
A = [4  1]
    [0  3]

det(A - λI) = det([4-λ    1  ])
                 [0    3-λ]
            = (4-λ)(3-λ)
            = λ² - 7λ + 12
            = (λ-3)(λ-4)

λ₁ = 3, λ₂ = 4
```

### 연습 2: 고유벡터 (λ = 3)
```
(A - 3I)v = 0

[1  1] [v₁]   [0]
[0  0] [v₂] = [0]

v₁ + v₂ = 0

v = [1]  또는 [t]  (t ≠ 0)
    [-1]     [-t]
```

### 연습 3: 확인
```
A = [4  1]    v = [1]
    [0  3]        [-1]

Av = [4×1 + 1×(-1)]  = [3]    = 3v ✓
     [0×1 + 3×(-1)]    [-3]
```

---

## 🔗 LLM 연결점

### 1. 주성분 분석 (PCA)
```
데이터의 공분산 행렬 C의 고유벡터
→ 데이터 분산이 최대인 방향
→ 차원 축소에 활용
```

### 2. 특이값 분해 (SVD)
```
A = UΣV^T

U, V의 열: 고유벡터
Σ: 특이값 (고유값의 제곱근)

임베딩 압축, 행렬 분해에 활용
```

### 3. PageRank 알고리즘
```
Google의 PageRank:
웹 그래프의 전이 행렬의
가장 큰 고유값에 대응하는 고유벡터

→ 중요도 점수
```

---

## ✅ 체크포인트

- [ ] **고유값과 고유벡터의 정의를 이해했나요?**

- [ ] **Av = λv의 의미를 설명할 수 있나요?**

- [ ] **2×2 행렬의 고유값을 계산할 수 있나요?**

- [ ] **PCA와의 연결을 이해했나요?**

---

## 🎓 핵심 요약

1. **정의**: Av = λv
2. **고유값**: det(A - λI) = 0
3. **고유벡터**: (A - λI)v = 0
4. **대각화**: A = PDP⁻¹
5. **PCA**: 공분산 행렬의 고유벡터 = 주성분

### 다음 학습
- **Day 19**: 행렬식과 노름

---

**수고하셨습니다!** 🎉

**고유벡터는 데이터 분석의 핵심 도구입니다!**
