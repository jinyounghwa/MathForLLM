# Day 11: 벡터의 길이와 거리 (1시간)

## 📚 학습 목표
- L2 노름(유클리드 노름)의 의미 완벽히 이해하기
- 벡터 간 거리 계산하기
- 피타고라스 정리와의 연결 이해하기
- LLM 임베딩 거리 계산의 기초 다지기

---

## 🎯 강의 주제
**"L2 노름과 피타고라스 정리"**

---

## 📖 핵심 개념

### 1. 벡터의 길이 (Norm)

#### 1.1 L2 노름 (Euclidean Norm)
**가장 일반적인 벡터의 길이**

```
||v|| = ||v||₂ = √(v₁² + v₂² + ... + vₙ²)
```

**2D 예시**:
```
v = [3, 4]
||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```

**3D 예시**:
```
v = [1, 2, 2]
||v|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3
```

#### 1.2 피타고라스 정리와의 연결
**2D 벡터**:
```
v = [a, b]

     *
    /|
  v/ |b
  /  |
 /___|
   a

||v||² = a² + b²  (피타고라스!)
||v|| = √(a² + b²)
```

**3D 벡터**: 일반화된 피타고라스
```
v = [a, b, c]
||v||² = a² + b² + c²
```

---

### 2. 다양한 노름 (Norms)

#### 2.1 L1 노름 (Manhattan Norm)
**절댓값의 합**

```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

**예시**:
```
v = [3, 4]
||v||₁ = |3| + |4| = 7

(격자를 따라 이동하는 거리)
```

**시각화**:
```
  4 ↑ → → → *
    | | | | |
  3 | | | | |
    | | | | |
  2 | | | | |
    | | | | |
  1 | | | | |
    | | | | |
  0 *-------→ 3

L1 거리 = 3 + 4 = 7
L2 거리 = 5
```

#### 2.2 L∞ 노름 (Maximum Norm)
**가장 큰 절댓값**

```
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

**예시**:
```
v = [3, 4]
||v||∞ = max(3, 4) = 4
```

#### 2.3 일반 Lp 노름
```
||v||ₚ = (|v₁|ᵖ + |v₂|ᵖ + ... + |vₙ|ᵖ)^(1/p)
```

- p = 1: L1 노름
- p = 2: L2 노름 (가장 일반적)
- p = ∞: L∞ 노름

---

### 3. 벡터 간 거리 (Distance)

#### 3.1 유클리드 거리
**두 점 사이의 직선 거리**

```
d(u, v) = ||u - v|| = √((u₁-v₁)² + (u₂-v₂)² + ... + (uₙ-vₙ)²)
```

**예시**:
```
u = [1, 2]
v = [4, 6]

d(u, v) = ||[1-4, 2-6]||
        = ||[-3, -4]||
        = √(9 + 16)
        = 5
```

#### 3.2 맨해튼 거리 (Manhattan Distance)
```
d₁(u, v) = ||u - v||₁ = |u₁-v₁| + |u₂-v₂| + ... + |uₙ-vₙ|
```

**예시**:
```
u = [1, 2]
v = [4, 6]

d₁(u, v) = |1-4| + |2-6| = 3 + 4 = 7
```

---

### 4. 거리의 성질

#### 4.1 거리 함수의 공리
모든 거리 함수 d는 다음을 만족:

**1. 비음수성 (Non-negativity)**:
```
d(u, v) ≥ 0
d(u, v) = 0 ⟺ u = v
```

**2. 대칭성 (Symmetry)**:
```
d(u, v) = d(v, u)
```

**3. 삼각 부등식 (Triangle Inequality)**:
```
d(u, w) ≤ d(u, v) + d(v, w)
```

**시각적 의미**:
```
u → v → w 경로가
u → w 직행보다 짧을 수 없다
```

---

## 💻 Python 실습

### 실습 1: 다양한 노름 계산
```python
import numpy as np

def compute_norms(v):
    """벡터의 다양한 노름 계산"""
    l1 = np.sum(np.abs(v))
    l2 = np.sqrt(np.sum(v**2))
    l_inf = np.max(np.abs(v))

    return l1, l2, l_inf

# 예시 벡터
v = np.array([3, 4])

print("=== 벡터의 노름 ===")
print(f"v = {v}")
print()

l1, l2, l_inf = compute_norms(v)
print(f"L1 노름:  ||v||₁ = {l1}")
print(f"L2 노름:  ||v||₂ = {l2:.4f}")
print(f"L∞ 노름: ||v||∞ = {l_inf}")
print()

# NumPy의 norm 함수
print("NumPy linalg.norm:")
print(f"L1: {np.linalg.norm(v, ord=1)}")
print(f"L2: {np.linalg.norm(v, ord=2):.4f}")
print(f"L∞: {np.linalg.norm(v, ord=np.inf)}")
```

### 실습 2: 고차원 벡터
```python
import numpy as np

# 고차원 벡터 (LLM 임베딩 시뮬레이션)
np.random.seed(42)
embedding_dim = 512

# 두 개의 임베딩 벡터
emb1 = np.random.randn(embedding_dim)
emb2 = np.random.randn(embedding_dim)

print("=== 고차원 벡터 노름 ===")
print(f"임베딩 차원: {embedding_dim}")
print()

# 노름 계산
norm1 = np.linalg.norm(emb1)
norm2 = np.linalg.norm(emb2)

print(f"||emb1|| = {norm1:.4f}")
print(f"||emb2|| = {norm2:.4f}")
print()

# 정규화된 벡터
emb1_normalized = emb1 / norm1
emb2_normalized = emb2 / norm2

print("정규화 후:")
print(f"||emb1_normalized|| = {np.linalg.norm(emb1_normalized):.4f}")
print(f"||emb2_normalized|| = {np.linalg.norm(emb2_normalized):.4f}")
print()

print("✅ 정규화하면 노름이 1이 됩니다!")
```

### 실습 3: 벡터 간 거리
```python
import numpy as np
import matplotlib.pyplot as plt

# 2D 점들
points = {
    'A': np.array([1, 2]),
    'B': np.array([4, 6]),
    'C': np.array([7, 3]),
    'D': np.array([3, 1])
}

def euclidean_distance(u, v):
    """유클리드 거리"""
    return np.linalg.norm(u - v)

def manhattan_distance(u, v):
    """맨해튼 거리"""
    return np.sum(np.abs(u - v))

# 거리 계산
print("=== 벡터 간 거리 ===")
point_pairs = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', 'D')]

for p1, p2 in point_pairs:
    u, v = points[p1], points[p2]
    euc_dist = euclidean_distance(u, v)
    man_dist = manhattan_distance(u, v)

    print(f"{p1}{u} ↔ {p2}{v}")
    print(f"  유클리드 거리: {euc_dist:.4f}")
    print(f"  맨해튼 거리:   {man_dist:.4f}")
    print()

# 시각화
plt.figure(figsize=(10, 10))
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# 점 그리기
for name, point in points.items():
    plt.plot(point[0], point[1], 'ro', markersize=10)
    plt.text(point[0]+0.2, point[1]+0.2, name, fontsize=14, fontweight='bold')

# A-B 거리 시각화
A, B = points['A'], points['B']
plt.plot([A[0], B[0]], [A[1], B[1]], 'b-', linewidth=2, label='유클리드')

# 맨해튼 거리 (격자)
plt.plot([A[0], B[0]], [A[1], A[1]], 'r--', linewidth=2, alpha=0.7)
plt.plot([B[0], B[0]], [A[1], B[1]], 'r--', linewidth=2, alpha=0.7, label='맨해튼')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('벡터 간 거리', fontsize=14)
plt.legend()
plt.xlim(0, 8)
plt.ylim(0, 7)
plt.tight_layout()
plt.savefig('vector_distances.png', dpi=150, bbox_inches='tight')
print("거리 시각화 저장 완료!")
```

### 실습 4: LLM 임베딩 거리
```python
import numpy as np

# 단어 임베딩 시뮬레이션 (간단한 예시)
np.random.seed(42)
dim = 128

# 단어들의 임베딩
embeddings = {
    '사과': np.random.randn(dim) + np.array([1, 0.5] + [0]*(dim-2)),
    '배':   np.random.randn(dim) + np.array([1, 0.4] + [0]*(dim-2)),
    '과일': np.random.randn(dim) + np.array([0.9, 0.6] + [0]*(dim-2)),
    '자동차': np.random.randn(dim) + np.array([-1, -0.5] + [0]*(dim-2)),
    '버스': np.random.randn(dim) + np.array([-0.9, -0.6] + [0]*(dim-2))
}

print("=== LLM 임베딩 거리 ===")
print(f"임베딩 차원: {dim}")
print()

# 단어 쌍들 간의 거리
word_pairs = [
    ('사과', '배'),
    ('사과', '과일'),
    ('사과', '자동차'),
    ('자동차', '버스'),
    ('사과', '버스')
]

print("단어 쌍 간 유클리드 거리:")
for w1, w2 in word_pairs:
    emb1 = embeddings[w1]
    emb2 = embeddings[w2]
    dist = np.linalg.norm(emb1 - emb2)
    print(f"  '{w1}' ↔ '{w2}': {dist:.4f}")

print()
print("✅ 의미가 비슷한 단어일수록 거리가 가깝습니다!")
print("   (사과-배 < 사과-자동차)")
```

### 실습 5: 삼각 부등식 검증
```python
import numpy as np

# 세 점
u = np.array([0, 0])
v = np.array([3, 4])
w = np.array([6, 0])

# 거리 계산
d_uv = np.linalg.norm(v - u)
d_vw = np.linalg.norm(w - v)
d_uw = np.linalg.norm(w - u)

print("=== 삼각 부등식 검증 ===")
print(f"u = {u}")
print(f"v = {v}")
print(f"w = {w}")
print()

print(f"d(u, v) = {d_uv:.4f}")
print(f"d(v, w) = {d_vw:.4f}")
print(f"d(u, w) = {d_uw:.4f}")
print()

print(f"d(u, v) + d(v, w) = {d_uv + d_vw:.4f}")
print(f"d(u, w) = {d_uw:.4f}")
print()

if d_uw <= d_uv + d_vw:
    print("✓ d(u, w) ≤ d(u, v) + d(v, w)")
    print("  삼각 부등식 성립!")
else:
    print("✗ 삼각 부등식 불성립 (이상함)")

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.grid(True, alpha=0.3)

# 점들
plt.plot(u[0], u[1], 'ro', markersize=12, label='u')
plt.plot(v[0], v[1], 'go', markersize=12, label='v')
plt.plot(w[0], w[1], 'bo', markersize=12, label='w')

# 선들
plt.plot([u[0], v[0]], [u[1], v[1]], 'r-', linewidth=2, label=f'd(u,v)={d_uv:.2f}')
plt.plot([v[0], w[0]], [v[1], w[1]], 'g-', linewidth=2, label=f'd(v,w)={d_vw:.2f}')
plt.plot([u[0], w[0]], [u[1], w[1]], 'b--', linewidth=2, label=f'd(u,w)={d_uw:.2f}')

plt.text(u[0]-0.5, u[1]-0.5, 'u', fontsize=14)
plt.text(v[0]+0.3, v[1]+0.3, 'v', fontsize=14)
plt.text(w[0]+0.3, w[1]-0.5, 'w', fontsize=14)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('삼각 부등식', fontsize=14)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('triangle_inequality.png', dpi=150, bbox_inches='tight')
print("\n삼각 부등식 시각화 저장 완료!")
```

---

## ✍️ 손 계산 연습

### 연습 1: L2 노름
다음 벡터의 L2 노름을 계산하세요:

1. v = [5, 12]
   ```
   ||v|| = √(5² + 12²) = √(25 + 144) = √169 = 13
   ```

2. v = [1, 2, 2]
   ```
   ||v|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3
   ```

### 연습 2: L1 노름
v = [3, -4]의 L1 노름:
```
||v||₁ = |3| + |-4| = 3 + 4 = 7
```

### 연습 3: 유클리드 거리
u = [1, 2], v = [4, 6] 사이의 거리:
```
d(u, v) = ||v - u||
        = ||[3, 4]||
        = √(9 + 16)
        = 5
```

### 연습 4: 삼각 부등식
u = [0, 0], v = [3, 0], w = [3, 4]일 때, 삼각 부등식 확인:
```
d(u, v) = 3
d(v, w) = 4
d(u, w) = √(9 + 16) = 5

d(u, w) = 5 ≤ 3 + 4 = 7 ✓
```

---

## 🔗 LLM 연결점

### 1. 임베딩 유사도 검색
```python
# 쿼리 임베딩
query_emb = [0.1, 0.2, ..., 0.5]  # 512차원

# 데이터베이스의 문서 임베딩들
docs_emb = [
    [0.12, 0.19, ..., 0.48],  # Doc 1
    [0.08, 0.25, ..., 0.52],  # Doc 2
    ...
]

# 가장 가까운 문서 찾기
distances = [||query_emb - doc_emb|| for doc_emb in docs_emb]
nearest_doc = argmin(distances)
```

### 2. RAG (Retrieval-Augmented Generation)
```
1. 쿼리를 임베딩으로 변환
2. 벡터 DB에서 거리가 가까운 문서 검색
3. 검색된 문서를 컨텍스트로 사용
4. LLM이 답변 생성
```

거리 계산이 핵심!

### 3. 정규화의 이유
```python
# 정규화하지 않으면
# 문서 길이에 따라 노름이 달라짐

# 정규화하면
# 순수하게 방향(의미)만 비교 가능
emb_normalized = emb / ||emb||
```

---

## ✅ 체크포인트

- [ ] **L2 노름을 계산할 수 있나요?**

- [ ] **피타고라스 정리와의 연결을 이해했나요?**

- [ ] **유클리드 거리와 맨해튼 거리의 차이를 설명할 수 있나요?**

- [ ] **삼각 부등식의 의미를 이해했나요?**

- [ ] **LLM 임베딩에서 거리의 역할을 설명할 수 있나요?**

---

## 🎓 핵심 요약

1. **L2 노름**: ||v|| = √(v₁² + ... + vₙ²)
2. **피타고라스 정리**: 2D, 3D, 고차원으로 일반화
3. **유클리드 거리**: d(u, v) = ||u - v||
4. **거리의 성질**: 비음수성, 대칭성, 삼각 부등식
5. **LLM 응용**: 임베딩 유사도, RAG

### 다음 학습
- **Day 12**: 내적 (Dot Product)
  - Attention 메커니즘의 핵심!

---

**수고하셨습니다!** 🎉
