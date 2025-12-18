# Day 12: 내적 (Dot Product) (1.5시간)

## 📚 학습 목표
- 내적의 정의와 계산 방법 완벽히 이해하기
- 내적의 기하학적 의미 (투영, 각도) 파악하기
- **Attention 메커니즘과의 연결 이해하기** ⭐
- 코사인 유사도의 기초 다지기

---

## 🎯 강의 주제
**"두 벡터가 얼마나 같은 방향인가?"**

---

## 📖 핵심 개념

### 1. 내적의 정의

#### 1.1 대수적 정의
**성분별 곱셈의 합**

```
a⃗ · b⃗ = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**2D 예시**:
```
a⃗ = [2, 3]
b⃗ = [4, 1]

a⃗ · b⃗ = (2)(4) + (3)(1) = 8 + 3 = 11
```

**3D 예시**:
```
a⃗ = [1, 2, 3]
b⃗ = [4, 5, 6]

a⃗ · b⃗ = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32
```

#### 1.2 행렬 표현
**행 벡터 × 열 벡터**

```
a⃗ · b⃗ = a⃗ᵀ b⃗ = [a₁ a₂ ... aₙ] [b₁]
                                 [b₂]
                                 [...]
                                 [bₙ]
```

---

### 2. 내적의 기하학적 의미 ⭐

#### 2.1 기하학적 정의
**내적 = 크기 × 크기 × 각도의 코사인**

```
a⃗ · b⃗ = ||a⃗|| × ||b⃗|| × cos(θ)
```

여기서 θ는 두 벡터 사이의 각도

**증명 (2D)**:
```
a⃗ = [a₁, a₂], b⃗ = [b₁, b₂]

코사인 법칙을 사용하면:
||a⃗ - b⃗||² = ||a⃗||² + ||b⃗||² - 2||a⃗||||b⃗||cos(θ)

전개하면:
(a₁-b₁)² + (a₂-b₂)² = a₁² + a₂² + b₁² + b₂² - 2||a⃗||||b⃗||cos(θ)
a₁² - 2a₁b₁ + b₁² + a₂² - 2a₂b₂ + b₂² = a₁² + a₂² + b₁² + b₂² - 2||a⃗||||b⃗||cos(θ)
-2(a₁b₁ + a₂b₂) = -2||a⃗||||b⃗||cos(θ)
a₁b₁ + a₂b₂ = ||a⃗||||b⃗||cos(θ)

∴ a⃗ · b⃗ = ||a⃗||||b⃗||cos(θ) ✓
```

#### 2.2 각도에 따른 내적 값

**같은 방향 (θ = 0°)**:
```
cos(0°) = 1
a⃗ · b⃗ = ||a⃗|| × ||b⃗|| × 1 > 0 (최대값)
```

**수직 (θ = 90°)**:
```
cos(90°) = 0
a⃗ · b⃗ = 0  (직교)
```

**반대 방향 (θ = 180°)**:
```
cos(180°) = -1
a⃗ · b⃗ = ||a⃗|| × ||b⃗|| × (-1) < 0 (최소값)
```

**시각화**:
```
       a⃗
      ↗
     /θ=45° → a⃗ · b⃗ > 0 (양수)
    ↗
   b⃗

       a⃗
      ↗
     /θ=90° → a⃗ · b⃗ = 0
    →
   b⃗

       a⃗
      ↗
     /θ=135° → a⃗ · b⃗ < 0 (음수)
    ↙
   b⃗
```

---

### 3. 투영 (Projection)

#### 3.1 스칼라 투영
**a⃗를 b⃗ 방향으로 투영한 길이**

```
proj_b⃗(a⃗) = ||a⃗|| cos(θ) = (a⃗ · b⃗) / ||b⃗||
```

**시각화**:
```
    a⃗
   /|
  / |
 /  |h
/θ  |
----+----→ b⃗
  proj
```

#### 3.2 벡터 투영
**a⃗를 b⃗ 방향으로 투영한 벡터**

```
proj_b⃗(a⃗) = ((a⃗ · b⃗) / ||b⃗||²) b⃗
           = ((a⃗ · b̂)) b̂  (b̂는 단위 벡터)
```

**예시**:
```
a⃗ = [3, 4]
b⃗ = [1, 0]  (x축 방향)

proj_b⃗(a⃗) = ((3×1 + 4×0) / 1²) [1, 0]
           = 3 [1, 0]
           = [3, 0]
```

---

### 4. 직교성 (Orthogonality)

#### 4.1 직교 벡터
**내적이 0인 벡터들**

```
a⃗ ⊥ b⃗ ⟺ a⃗ · b⃗ = 0
```

**예시**:
```
a⃗ = [1, 0]
b⃗ = [0, 1]

a⃗ · b⃗ = 1×0 + 0×1 = 0 → 직교!
```

**직교 기저**:
```
e⃗₁ = [1, 0, 0]
e⃗₂ = [0, 1, 0]
e⃗₃ = [0, 0, 1]

e⃗ᵢ · e⃗ⱼ = 0 (i ≠ j)
```

---

### 5. 내적의 성질

**1. 교환법칙**:
```
a⃗ · b⃗ = b⃗ · a⃗
```

**2. 분배법칙**:
```
a⃗ · (b⃗ + c⃗) = a⃗ · b⃗ + a⃗ · c⃗
```

**3. 스칼라배**:
```
(ka⃗) · b⃗ = k(a⃗ · b⃗)
```

**4. 자기 자신과의 내적**:
```
a⃗ · a⃗ = ||a⃗||²
```

---

## 💻 Python 실습

### 실습 1: 내적 계산
```python
import numpy as np

# 벡터 정의
a = np.array([2, 3])
b = np.array([4, 1])

print("=== 내적 계산 ===")
print(f"a⃗ = {a}")
print(f"b⃗ = {b}")
print()

# 방법 1: 수동 계산
dot_manual = a[0]*b[0] + a[1]*b[1]
print(f"수동 계산: a⃗ · b⃗ = {dot_manual}")

# 방법 2: NumPy dot
dot_numpy = np.dot(a, b)
print(f"np.dot(a, b) = {dot_numpy}")

# 방법 3: @ 연산자
dot_at = a @ b
print(f"a @ b = {dot_at}")

# 방법 4: sum
dot_sum = np.sum(a * b)
print(f"np.sum(a * b) = {dot_sum}")

print(f"\n모두 같은 결과: {dot_manual}")
```

### 실습 2: 각도 계산
```python
import numpy as np

def angle_between(a, b, degrees=True):
    """두 벡터 사이의 각도 계산"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    cos_theta = dot / (norm_a * norm_b)
    # 수치 오차로 인한 범위 초과 방지
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta_rad = np.arccos(cos_theta)

    if degrees:
        return np.degrees(theta_rad)
    return theta_rad

# 다양한 벡터 쌍
pairs = [
    ("같은 방향", np.array([1, 0]), np.array([2, 0])),
    ("수직", np.array([1, 0]), np.array([0, 1])),
    ("반대 방향", np.array([1, 0]), np.array([-1, 0])),
    ("45도", np.array([1, 0]), np.array([1, 1])),
    ("임의", np.array([3, 4]), np.array([1, 2]))
]

print("=== 벡터 간 각도 ===\n")

for name, a, b in pairs:
    dot = np.dot(a, b)
    angle = angle_between(a, b)

    print(f"{name}:")
    print(f"  a⃗ = {a}, b⃗ = {b}")
    print(f"  a⃗ · b⃗ = {dot:.2f}")
    print(f"  각도 θ = {angle:.2f}°")
    print()
```

### 실습 3: 투영
```python
import numpy as np
import matplotlib.pyplot as plt

def scalar_projection(a, b):
    """스칼라 투영"""
    return np.dot(a, b) / np.linalg.norm(b)

def vector_projection(a, b):
    """벡터 투영"""
    return (np.dot(a, b) / np.dot(b, b)) * b

# 벡터 정의
a = np.array([3, 2])
b = np.array([4, 1])

# 투영 계산
scalar_proj = scalar_projection(a, b)
vector_proj = vector_projection(a, b)

print("=== 벡터 투영 ===")
print(f"a⃗ = {a}")
print(f"b⃗ = {b}")
print()
print(f"스칼라 투영: {scalar_proj:.4f}")
print(f"벡터 투영: {vector_proj}")
print()

# 시각화
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# 원본 벡터
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.01, label='a⃗')
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.01, label='b⃗')

# 투영 벡터
ax.quiver(0, 0, vector_proj[0], vector_proj[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.015, label='proj_b⃗(a⃗)')

# 수직선 (a에서 투영까지)
ax.plot([a[0], vector_proj[0]], [a[1], vector_proj[1]], 'k--', alpha=0.5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('벡터 투영', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('vector_projection.png', dpi=150, bbox_inches='tight')
print("투영 시각화 저장 완료!")
```

### 실습 4: Attention 메커니즘 시뮬레이션
```python
import numpy as np

def attention_scores(query, keys):
    """
    Query와 Keys 사이의 Attention 점수 계산
    점수 = 내적 (Scaled Dot-Product는 나중에)
    """
    scores = []
    for key in keys:
        score = np.dot(query, key)
        scores.append(score)
    return np.array(scores)

# 시뮬레이션: 단어 임베딩
dim = 8  # 간단한 예시

# Query: "고양이"
query = np.array([1.0, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])

# Keys: 문장 내 다른 단어들
keys = {
    "귀여운": np.array([0.9, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]),
    "강아지": np.array([0.8, 0.9, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]),
    "자동차": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.2, 0.1]),
    "운전": np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.7, 0.3, 0.2])
}

print("=== Attention 메커니즘 (내적 기반) ===")
print(f"Query (고양이): {query}")
print()

print("Keys와의 내적 (유사도):")
for word, key in keys.items():
    score = np.dot(query, key)
    print(f"  '{word}': {score:.4f}")

print()
print("✅ 내적이 클수록 관련성이 높음!")
print("   '고양이'는 '귀여운', '강아지'와 관련 높음")
print("   '자동차', '운전'과는 관련 낮음")
```

### 실습 5: 직교 벡터
```python
import numpy as np
import matplotlib.pyplot as plt

# 직교 벡터 생성
def generate_orthogonal_2d(v):
    """2D 벡터에 직교하는 벡터 생성"""
    return np.array([-v[1], v[0]])

# 원본 벡터
v = np.array([3, 2])
v_orth = generate_orthogonal_2d(v)

print("=== 직교 벡터 ===")
print(f"v = {v}")
print(f"v_orth = {v_orth}")
print(f"v · v_orth = {np.dot(v, v_orth)}")
print()

# 3D 직교 기저
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

print("3D 표준 기저:")
print(f"e₁ · e₂ = {np.dot(e1, e2)}")
print(f"e₂ · e₃ = {np.dot(e2, e3)}")
print(f"e₃ · e₁ = {np.dot(e3, e1)}")
print("모두 0 → 서로 직교!")

# 시각화
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# 벡터 그리기
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.015, label='v')
ax.quiver(0, 0, v_orth[0], v_orth[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.015, label='v_orth (v ⊥ v_orth)')

# 직각 표시
from matplotlib.patches import Rectangle
rect_size = 0.5
rect = Rectangle((0, 0), rect_size, rect_size,
                  fill=False, edgecolor='green', linewidth=2)
ax.add_patch(rect)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('직교 벡터 (v · v_orth = 0)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('orthogonal_vectors.png', dpi=150, bbox_inches='tight')
print("\n직교 벡터 시각화 저장 완료!")
```

---

## ✍️ 손 계산 연습

### 연습 1: 내적 계산
```
a⃗ = [1, 2, 3]
b⃗ = [4, 5, 6]

a⃗ · b⃗ = (1)(4) + (2)(5) + (3)(6)
       = 4 + 10 + 18
       = 32
```

### 연습 2: 각도 계산
```
a⃗ = [1, 0]
b⃗ = [1, 1]

a⃗ · b⃗ = 1
||a⃗|| = 1
||b⃗|| = √2

cos(θ) = 1 / (1 × √2) = 1/√2 = √2/2
θ = 45°
```

### 연습 3: 직교 확인
```
a⃗ = [3, -2]
b⃗ = [2, 3]

a⃗ · b⃗ = (3)(2) + (-2)(3) = 6 - 6 = 0
→ 직교! ✓
```

### 연습 4: 투영
```
a⃗ = [3, 4]를 x축 (b⃗ = [1, 0])에 투영:

proj_b⃗(a⃗) = ((a⃗ · b⃗) / ||b⃗||²) b⃗
           = (3 / 1) [1, 0]
           = [3, 0]
```

---

## 🔗 LLM 연결점

### 1. Attention 메커니즘의 핵심
**Query와 Key의 내적**:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

QKᵀ = 각 Query와 Key 사이의 내적!
```

**의미**:
- 내적 크다 = Query와 Key가 유사 = 높은 attention 점수
- 내적 작다 = 관련 없음 = 낮은 attention 점수

### 2. 코사인 유사도 (다음 강의 예고)
```
similarity = (a⃗ · b⃗) / (||a⃗|| × ||b⃗||) = cos(θ)
```

RAG 시스템의 핵심!

### 3. 선형 층 (Linear Layer)
```
y = Wx + b

y_i = W_i · x  (i번째 행과 x의 내적)
```

신경망의 모든 선형 변환 = 내적의 연속!

---

## ✅ 체크포인트

- [ ] **내적을 계산할 수 있나요?**

- [ ] **a⃗ · b⃗ = ||a⃗||||b⃗||cos(θ)의 의미를 이해했나요?**

- [ ] **직교 벡터를 찾을 수 있나요?**

- [ ] **투영의 개념을 이해했나요?**

- [ ] **Attention 메커니즘에서 내적의 역할을 설명할 수 있나요?**

---

## 🎓 핵심 요약

1. **내적**: a⃗ · b⃗ = Σ aᵢbᵢ
2. **기하학**: a⃗ · b⃗ = ||a⃗||||b⃗||cos(θ)
3. **직교**: a⃗ · b⃗ = 0 ⟺ a⃗ ⊥ b⃗
4. **투영**: proj_b⃗(a⃗) = ((a⃗ · b⃗) / ||b⃗||²) b⃗
5. **Attention**: QKᵀ = Query와 Key의 내적

### 다음 학습
- **Day 13**: 정규화 (Normalization)
  - 코사인 유사도
  - RAG 시스템

---

**수고하셨습니다!** 🎉

**내적은 LLM의 가장 중요한 연산입니다!**
