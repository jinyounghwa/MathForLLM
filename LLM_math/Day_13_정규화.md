# Day 13: 정규화 (Normalization) (1.5시간)

## 📚 학습 목표
- 벡터 정규화의 개념과 방법 이해하기
- 코사인 유사도 완벽히 이해하기
- RAG 시스템에서의 정규화 활용 파악하기
- L2 정규화와 Batch Normalization 구별하기

---

## 🎯 강의 주제
**"벡터를 길이 1로 만들기"**

---

## 📖 핵심 개념

### 1. 벡터 정규화 (Vector Normalization)

#### 1.1 정의
**벡터를 그 크기로 나누어 길이를 1로 만들기**

```
v̂ = v / ||v||
```

**결과**:
- 방향은 유지
- 크기는 1

**예시**:
```
v = [3, 4]
||v|| = 5

v̂ = [3/5, 4/5] = [0.6, 0.8]
||v̂|| = √(0.36 + 0.64) = 1 ✓
```

#### 1.2 단위 벡터 (Unit Vector)
**정규화된 벡터 = 단위 벡터**

```
||v̂|| = 1
```

**표준 기저 벡터도 단위 벡터**:
```
e⃗ₓ = [1, 0, 0]  → ||e⃗ₓ|| = 1
e⃗ᵧ = [0, 1, 0]  → ||e⃗ᵧ|| = 1
e⃗_z = [0, 0, 1]  → ||e⃗_z|| = 1
```

---

### 2. 코사인 유사도 (Cosine Similarity) ⭐

#### 2.1 정의
**두 벡터가 가리키는 방향의 유사도**

```
similarity = cos(θ) = (a⃗ · b⃗) / (||a⃗|| × ||b⃗||)
```

**정규화된 벡터로 표현**:
```
similarity = â · b̂
```

**범위**: -1 ~ 1
- 1: 완전히 같은 방향 (θ = 0°)
- 0: 직교 (θ = 90°)
- -1: 완전히 반대 방향 (θ = 180°)

#### 2.2 왜 코사인을 사용하는가?

**문제**: 유클리드 거리는 벡터 크기에 민감
```
v₁ = [1, 0]
v₂ = [2, 0]  (같은 방향, 2배 길이)

거리 = ||v₂ - v₁|| = 1 (다르다고 판단)
```

**해결**: 코사인 유사도는 방향만 비교
```
cos(θ) = 1 (같은 방향!)
```

#### 2.3 예시
```
a⃗ = [1, 2, 3]
b⃗ = [2, 4, 6]  (a의 2배)

cos(θ) = (2 + 8 + 18) / (√14 × √56)
       = 28 / 28
       = 1 (완전히 같은 방향)

c⃗ = [1, 0, 0]
cos(θ_ac) = 1 / √14 ≈ 0.27 (다른 방향)
```

---

### 3. L2 정규화 (L2 Normalization)

#### 3.1 정의
**각 벡터를 L2 노름으로 나누기**

```
x_normalized = x / ||x||₂
```

**특징**:
- 각 샘플을 독립적으로 정규화
- 방향만 중요한 경우 사용

#### 3.2 LLM 임베딩에서의 사용
```python
# 임베딩 벡터
embedding = [0.1, 0.2, ..., 0.5]  # 512차원

# L2 정규화
norm = ||embedding||
embedding_normalized = embedding / norm

# 이제 ||embedding_normalized|| = 1
```

**이점**:
1. 코사인 유사도 계산 단순화
   ```
   similarity = emb1_norm · emb2_norm
   ```

2. 벡터 크기 무시, 순수 방향 비교

---

### 4. RAG에서의 정규화

#### 4.1 RAG 파이프라인
```
1. 문서를 임베딩으로 변환
2. 임베딩을 L2 정규화
3. 벡터 DB에 저장
4. 쿼리를 임베딩으로 변환
5. 쿼리 임베딩 L2 정규화
6. 코사인 유사도로 가장 유사한 문서 검색
```

#### 4.2 예시
```python
# 문서 임베딩
doc1 = [0.1, 0.2, 0.3, ...]  # 길이: 1.5
doc2 = [0.2, 0.4, 0.6, ...]  # 길이: 3.0 (doc1의 2배)

# 정규화하지 않으면
# doc2가 더 길어서 유사도 계산에 영향

# 정규화하면
doc1_norm = doc1 / ||doc1||  # 길이: 1
doc2_norm = doc2 / ||doc2||  # 길이: 1

# 이제 순수하게 방향만 비교!
```

---

### 5. Batch Normalization vs Layer Normalization

#### 5.1 Batch Normalization
**배치 차원에서 정규화**

```
x_norm = (x - mean_batch) / std_batch
```

**특징**:
- 각 feature별로 배치 전체의 평균/표준편차 사용
- CNN에서 주로 사용

#### 5.2 Layer Normalization
**feature 차원에서 정규화**

```
x_norm = (x - mean_features) / std_features
```

**특징**:
- 각 샘플별로 모든 feature의 평균/표준편차 사용
- Transformer에서 사용

**비교**:
```
입력: [batch_size, features]

Batch Norm: 각 feature에 대해 batch 전체 정규화
Layer Norm: 각 샘플에 대해 feature 전체 정규화
```

---

## 💻 Python 실습

### 실습 1: 벡터 정규화
```python
import numpy as np

def normalize(v):
    """L2 정규화"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# 예시 벡터들
vectors = {
    'v1': np.array([3, 4]),
    'v2': np.array([1, 1, 1]),
    'v3': np.array([0, 0, 5]),
}

print("=== 벡터 정규화 ===\n")

for name, v in vectors.items():
    v_norm = normalize(v)
    norm_before = np.linalg.norm(v)
    norm_after = np.linalg.norm(v_norm)

    print(f"{name} = {v}")
    print(f"  정규화 전 크기: {norm_before:.4f}")
    print(f"  정규화 후: {v_norm}")
    print(f"  정규화 후 크기: {norm_after:.4f}")
    print()
```

### 실습 2: 코사인 유사도
```python
import numpy as np

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

def cosine_similarity_normalized(a_norm, b_norm):
    """정규화된 벡터의 코사인 유사도 (단순 내적)"""
    return np.dot(a_norm, b_norm)

# 예시 벡터
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # v1의 2배
v3 = np.array([1, 0, 0])  # 다른 방향

print("=== 코사인 유사도 ===\n")

print(f"v1 = {v1}")
print(f"v2 = {v2} (v1의 2배)")
print(f"v3 = {v3}")
print()

# 방법 1: 직접 계산
sim_12 = cosine_similarity(v1, v2)
sim_13 = cosine_similarity(v1, v3)

print("방법 1: 직접 계산")
print(f"  cos(v1, v2) = {sim_12:.4f}")
print(f"  cos(v1, v3) = {sim_13:.4f}")
print()

# 방법 2: 정규화 후 내적
v1_norm = v1 / np.linalg.norm(v1)
v2_norm = v2 / np.linalg.norm(v2)
v3_norm = v3 / np.linalg.norm(v3)

sim_12_norm = cosine_similarity_normalized(v1_norm, v2_norm)
sim_13_norm = cosine_similarity_normalized(v1_norm, v3_norm)

print("방법 2: 정규화 후 내적")
print(f"  v1_norm · v2_norm = {sim_12_norm:.4f}")
print(f"  v1_norm · v3_norm = {sim_13_norm:.4f}")
print()

print("✅ v2는 v1의 2배지만, 코사인 유사도는 1 (같은 방향!)")
```

### 실습 3: RAG 문서 검색 시뮬레이션
```python
import numpy as np

# 시뮬레이션: 간단한 문서 임베딩
np.random.seed(42)
dim = 128

# 문서 임베딩 (정규화되지 않음)
documents = {
    "Python 기초": np.random.randn(dim) + np.array([1, 1] + [0]*(dim-2)),
    "Python 고급": np.random.randn(dim) + np.array([1.2, 1.1] + [0]*(dim-2)),
    "Java 기초": np.random.randn(dim) + np.array([0.8, -0.5] + [0]*(dim-2)),
    "요리 레시피": np.random.randn(dim) + np.array([-1, 0.5] + [0]*(dim-2)),
    "운동 방법": np.random.randn(dim) + np.array([-0.5, 1] + [0]*(dim-2))
}

# 쿼리
query = "Python 프로그래밍 배우기"
query_emb = np.random.randn(dim) + np.array([1.1, 0.9] + [0]*(dim-2))

print("=== RAG 문서 검색 ===")
print(f"쿼리: '{query}'")
print(f"임베딩 차원: {dim}")
print()

# L2 정규화
query_norm = query_emb / np.linalg.norm(query_emb)
docs_norm = {name: emb / np.linalg.norm(emb) for name, emb in documents.items()}

# 코사인 유사도 계산
similarities = {}
for doc_name, doc_emb_norm in docs_norm.items():
    sim = np.dot(query_norm, doc_emb_norm)
    similarities[doc_name] = sim

# 정렬 (유사도 높은 순)
sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

print("검색 결과 (유사도 순):")
for rank, (doc_name, sim) in enumerate(sorted_docs, 1):
    print(f"  {rank}. {doc_name:20s}: {sim:.4f}")

print()
print("✅ 'Python' 관련 문서들이 상위에 랭크!")
```

### 실습 4: 정규화 시각화
```python
import numpy as np
import matplotlib.pyplot as plt

# 여러 벡터 생성
np.random.seed(42)
vectors = np.random.randn(10, 2) * 2

# 정규화
vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, vecs, title in zip(axes, [vectors, vectors_normalized],
                             ['Original Vectors', 'Normalized Vectors (Unit Circle)']):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # 벡터 그리기
    for v in vecs:
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  width=0.005, alpha=0.7)
        ax.plot(v[0], v[1], 'ro', markersize=5)

    # 정규화된 경우 단위원 그리기
    if 'Normalized' in title:
        circle = plt.Circle((0, 0), 1, fill=False, color='blue',
                            linewidth=2, linestyle='--', label='Unit Circle')
        ax.add_patch(circle)
        ax.legend()

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)

plt.tight_layout()
plt.savefig('normalization_visualization.png', dpi=150, bbox_inches='tight')
print("정규화 시각화 저장 완료!")
```

### 실습 5: Layer Normalization
```python
import numpy as np

def layer_norm(x, epsilon=1e-6):
    """Layer Normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

# 예시: [batch_size, features]
batch_size, features = 3, 4
x = np.random.randn(batch_size, features) * 2 + 5

print("=== Layer Normalization ===")
print(f"입력 shape: {x.shape}")
print(f"입력:\n{x}")
print()

# Layer Normalization 적용
x_norm = layer_norm(x)

print(f"정규화 후:\n{x_norm}")
print()

# 각 샘플의 통계
for i in range(batch_size):
    print(f"샘플 {i+1}:")
    print(f"  정규화 전: 평균={np.mean(x[i]):.4f}, 표준편차={np.std(x[i]):.4f}")
    print(f"  정규화 후: 평균={np.mean(x_norm[i]):.4f}, 표준편차={np.std(x_norm[i]):.4f}")
print()

print("✅ 정규화 후 각 샘플의 평균≈0, 표준편차≈1")
```

---

## ✍️ 손 계산 연습

### 연습 1: 벡터 정규화
```
v = [3, 4]

Step 1: 크기 계산
||v|| = √(9 + 16) = 5

Step 2: 정규화
v̂ = [3/5, 4/5] = [0.6, 0.8]

검증: ||v̂|| = √(0.36 + 0.64) = 1 ✓
```

### 연습 2: 코사인 유사도
```
a⃗ = [1, 2]
b⃗ = [2, 1]

Step 1: 내적
a⃗ · b⃗ = 2 + 2 = 4

Step 2: 크기
||a⃗|| = √5
||b⃗|| = √5

Step 3: 코사인 유사도
cos(θ) = 4 / (√5 × √5) = 4/5 = 0.8
```

### 연습 3: 정규화 후 내적
```
a⃗ = [3, 4], b⃗ = [5, 12]

Step 1: 정규화
â = [3/5, 4/5] = [0.6, 0.8]
b̂ = [5/13, 12/13] ≈ [0.385, 0.923]

Step 2: 내적 (= 코사인 유사도)
â · b̂ = 0.6×0.385 + 0.8×0.923
      = 0.231 + 0.738
      ≈ 0.97
```

---

## 🔗 LLM 연결점

### 1. Sentence Embedding + RAG
```python
# 1. 문서 임베딩 & 정규화
doc_emb = model.encode("문서 내용")
doc_emb_norm = doc_emb / ||doc_emb||

# 2. 저장
vector_db.store(doc_emb_norm)

# 3. 검색
query_emb = model.encode("질문")
query_emb_norm = query_emb / ||query_emb||

# 4. 코사인 유사도 (단순 내적)
scores = query_emb_norm @ doc_embs_norm.T
top_k = argmax(scores, k=5)
```

### 2. Attention에서의 정규화
```python
# Scaled Dot-Product Attention
scores = Q @ K.T / sqrt(d_k)

# √d_k로 나누는 이유:
# - 내적 값이 너무 커지는 것 방지
# - Softmax의 수치 안정성 향상
```

### 3. Layer Normalization in Transformer
```python
# Transformer의 각 서브층 후
x = x + sublayer(x)  # Residual Connection
x = LayerNorm(x)     # Layer Normalization
```

---

## ✅ 체크포인트

- [ ] **벡터를 정규화할 수 있나요?**

- [ ] **코사인 유사도를 계산할 수 있나요?**

- [ ] **정규화된 벡터의 코사인 유사도 = 내적임을 이해했나요?**

- [ ] **RAG에서 왜 정규화를 사용하는지 설명할 수 있나요?**

- [ ] **L2 정규화와 Layer Normalization의 차이를 이해했나요?**

---

## 🎓 핵심 요약

1. **정규화**: v̂ = v / ||v||, 크기를 1로
2. **코사인 유사도**: cos(θ) = (a⃗ · b⃗) / (||a⃗||||b⃗||)
3. **정규화 후**: cos(θ) = â · b̂ (단순 내적!)
4. **RAG**: 임베딩 정규화 → 코사인 유사도 검색
5. **Layer Norm**: Transformer의 핵심 구성 요소

### 다음 학습
- **Day 14**: 행렬과 행렬 곱셈
  - 신경망의 y = Wx + b

---

**수고하셨습니다!** 🎉

**정규화는 RAG와 임베딩 검색의 핵심입니다!**
