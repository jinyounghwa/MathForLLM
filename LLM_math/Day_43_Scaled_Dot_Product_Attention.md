# Day 43: Scaled Dot-Product Attention (2시간) ⭐

## 📚 학습 목표
- **Scaled Dot-Product Attention의 모든 단계 완벽히 이해하기**
- Q, K, V의 의미 파악하기
- √d_k로 나누는 이유 이해하기
- NumPy로 직접 구현하기

---

## 🎯 강의 주제
**"LLM의 심장 - Attention 메커니즘"**

---

## 📖 핵심 개념

### 1. Attention의 직관

**문제**:
```
문장: "The cat sat on the mat"

"it"이 무엇을 가리키는지 알려면?
→ 문맥(context)을 봐야 함!
```

**Attention의 답**:
```
각 단어가 다른 단어들에
얼마나 "주의(attention)"를 기울여야 하는지 계산
```

---

### 2. Query, Key, Value

**비유: 도서관 검색**
```
Query (질문):
"머신러닝 책을 찾습니다"

Key (책 제목):
"딥러닝", "알고리즘", "머신러닝 기초"

Value (책 내용):
실제 책

과정:
1. Query와 각 Key의 유사도 계산
2. 유사도를 가중치로
3. Value들의 가중 합 반환
```

**LLM에서**:
```
Query: "이 단어는 어떤 맥락인가?"
Key: "나는 이런 의미야"
Value: "내 정보를 전달해줄게"
```

---

### 3. Scaled Dot-Product Attention 공식

**공식**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**단계별**:
```
1. Score 계산: QK^T
   - Q와 K의 내적 (유사도)

2. Scaling: / √d_k
   - 차원이 클수록 내적이 커지는 것 방지
   - d_k: Key의 차원

3. Softmax: softmax(scaled scores)
   - 확률 분포로 변환
   - 합이 1

4. 가중합: × V
   - Value들을 가중 평균
```

---

### 4. 왜 √d_k로 나누나?

**이유**:
```
Q, K의 차원 d_k가 크면:
- 내적 값이 매우 커짐
- Softmax의 기울기 소실
- 학습 불안정

√d_k로 나누면:
- 분산이 1로 정규화
- Softmax가 안정적
```

**수식적 이해**:
```
Q, K ~ N(0, 1)인 d_k차원 벡터

Q·K의 분산 = d_k

Q·K / √d_k의 분산 = 1  ✓
```

---

### 5. 예제: 손 계산

**설정**:
```
단어 3개 (seq_len = 3)
차원 4 (d_k = 4)

Q = [1, 0, 1, 0]    # "cat"의 Query
    [0, 1, 0, 1]    # "sat"의 Query
    [1, 1, 0, 0]    # "mat"의 Query

K = [1, 0, 1, 0]    # "cat"의 Key
    [0, 1, 0, 1]    # "sat"의 Key
    [1, 1, 0, 0]    # "mat"의 Key

V = [1, 2, 3, 4]    # "cat"의 Value
    [5, 6, 7, 8]    # "sat"의 Value
    [9, 10, 11, 12] # "mat"의 Value
```

**1단계: QK^T**:
```
"cat" Query · "cat" Key = 1·1 + 0·0 + 1·1 + 0·0 = 2
"cat" Query · "sat" Key = 1·0 + 0·1 + 1·0 + 0·1 = 0
"cat" Query · "mat" Key = 1·1 + 0·1 + 1·0 + 0·0 = 1

scores = [2, 0, 1]
```

**2단계: Scaling**:
```
√d_k = √4 = 2
scaled = [2/2, 0/2, 1/2] = [1, 0, 0.5]
```

**3단계: Softmax**:
```
exp([1, 0, 0.5]) = [2.72, 1, 1.65]
sum = 5.37

softmax = [0.51, 0.19, 0.31]
```

**4단계: × V**:
```
output = 0.51[1,2,3,4] + 0.19[5,6,7,8] + 0.31[9,10,11,12]
       ≈ [4.2, 5.3, 6.4, 7.6]
```

---

## 💻 Python 실습

### 실습 1: Scaled Dot-Product Attention 구현
```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    Args:
        Q: Query (seq_len_q, d_k)
        K: Key (seq_len_k, d_k)
        V: Value (seq_len_v, d_v)
        mask: Optional mask

    Returns:
        output: (seq_len_q, d_v)
        attention_weights: (seq_len_q, seq_len_k)
    """
    d_k = K.shape[-1]

    # 1. Score 계산: QK^T
    scores = Q @ K.T  # (seq_len_q, seq_len_k)

    # 2. Scaling
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Mask (옵션)
    if mask is not None:
        scaled_scores += (mask * -1e9)

    # 4. Softmax
    attention_weights = np.exp(scaled_scores)
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)

    # 5. 가중합: × V
    output = attention_weights @ V  # (seq_len_q, d_v)

    return output, attention_weights


# 예제
print("=== Scaled Dot-Product Attention ===\n")

# 설정
seq_len = 3
d_k = 4
d_v = 4

Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0]], dtype=float)

K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0]], dtype=float)

V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]], dtype=float)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}\n")

# Attention 계산
output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights (각 행이 하나의 Query):")
print(attn_weights)
print()

print("Output (각 행이 하나의 출력):")
print(output)
print()

print("해석:")
print("- attn_weights[0]: 첫 번째 단어가 모든 단어에 주는 attention")
print("- 합이 1.0 (확률 분포)")
print("- output[0]: 첫 번째 단어의 맥락화된 표현")
```

### 실습 2: Self-Attention 시각화
```python
import numpy as np
import matplotlib.pyplot as plt

# 문장: "The cat sat on the mat"
words = ["The", "cat", "sat", "on", "the", "mat"]
n = len(words)

# 임의의 Q, K, V (실제로는 학습됨)
np.random.seed(42)
d_model = 8

Q = np.random.randn(n, d_model)
K = np.random.randn(n, d_model)
V = np.random.randn(n, d_model)

# Attention
output, attn_weights = scaled_dot_product_attention(Q, K, V)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Attention 히트맵
im = ax1.imshow(attn_weights, cmap='Blues', aspect='auto')
ax1.set_xticks(range(n))
ax1.set_yticks(range(n))
ax1.set_xticklabels(words)
ax1.set_yticklabels(words)
ax1.set_xlabel('Key (attending to)')
ax1.set_ylabel('Query (from)')
ax1.set_title('Attention Weights')

# 값 표시
for i in range(n):
    for j in range(n):
        text = ax1.text(j, i, f'{attn_weights[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax1)

# 특정 단어 ("cat")의 attention 분포
cat_idx = 1
ax2.bar(words, attn_weights[cat_idx])
ax2.set_xlabel('단어')
ax2.set_ylabel('Attention Weight')
ax2.set_title(f'"{words[cat_idx]}"이 각 단어에 주는 Attention')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('self_attention_visualization.png', dpi=150)
print("\nAttention 시각화 저장 완료!")
```

### 실습 3: Masking (Causal Attention)
```python
import numpy as np

def create_causal_mask(seq_len):
    """인과적 마스크 생성 (미래 단어 가리기)"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask  # 1 = mask, 0 = visible

# 예: GPT의 Causal Self-Attention
seq_len = 5
words = ["I", "love", "machine", "learning", "."]

# 마스크
mask = create_causal_mask(seq_len)

print("=== Causal (Masked) Attention ===\n")
print("마스크 (1 = 가려짐, 0 = 보임):")
print(mask)
print()

# 임의의 Q, K, V
np.random.seed(42)
Q = np.random.randn(seq_len, 4)
K = np.random.randn(seq_len, 4)
V = np.random.randn(seq_len, 4)

# Masked Attention
output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

print("Masked Attention Weights:")
for i, word in enumerate(words):
    visible_words = words[:i+1]
    print(f"{word:>10}: {attn_weights[i, :i+1]}")
    print(f"           (볼 수 있는 단어: {visible_words})\n")
```

### 실습 4: Scaling의 중요성
```python
import numpy as np

def attention_without_scaling(Q, K, V):
    """Scaling 없는 Attention"""
    scores = Q @ K.T
    attn_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights

# 다양한 차원에서 비교
dimensions = [4, 16, 64, 256]

print("=== Scaling의 효과 ===\n")

for d_k in dimensions:
    Q = np.random.randn(3, d_k)
    K = np.random.randn(3, d_k)
    V = np.random.randn(3, d_k)

    # Scaling 없음
    _, attn_no_scale = attention_without_scaling(Q, K, V)

    # Scaling 있음
    _, attn_scaled = scaled_dot_product_attention(Q, K, V)

    print(f"d_k = {d_k}:")
    print(f"  No scaling: max={attn_no_scale[0].max():.4f}, "
          f"min={attn_no_scale[0].min():.4f}")
    print(f"  Scaled:     max={attn_scaled[0].max():.4f}, "
          f"min={attn_scaled[0].min():.4f}")
    print()

print("→ 차원이 클수록 scaling이 중요함!")
```

---

## ✍️ 손 계산 연습

### 연습: 간단한 Attention
```
Q = [1, 1], K = [1, 0], V = [2, 3]
        [0, 1]      [0, 1]      [4, 5]

d_k = 2, √d_k = √2 ≈ 1.41

1. QK^T:
   [1,1]·[1,0] = 1,  [1,1]·[0,1] = 1
   [0,1]·[1,0] = 0,  [0,1]·[0,1] = 1

   scores = [[1, 1],
             [0, 1]]

2. Scaling:
   scaled = [[0.71, 0.71],
             [0, 0.71]]

3. Softmax (첫 행):
   exp([0.71, 0.71]) = [2.03, 2.03]
   softmax = [0.5, 0.5]

4. 첫 행 출력:
   0.5[2,3] + 0.5[4,5] = [3, 4]
```

---

## 🔗 LLM 연결점

### 1. GPT의 Causal Self-Attention
```python
# PyTorch (개념적)
scores = Q @ K.T / sqrt(d_k)
mask = causal_mask  # 미래 가리기
scores = scores.masked_fill(mask == 1, -1e9)
attn = softmax(scores, dim=-1)
output = attn @ V
```

### 2. BERT의 Bidirectional Self-Attention
```python
# 마스크 없음 (양방향)
scores = Q @ K.T / sqrt(d_k)
attn = softmax(scores, dim=-1)
output = attn @ V
```

---

## ✅ 체크포인트

- [ ] **Q, K, V의 역할을 설명할 수 있나요?**

- [ ] **√d_k로 나누는 이유를 이해했나요?**

- [ ] **Attention을 손으로 계산할 수 있나요?**

- [ ] **Causal Attention과 일반 Attention의 차이를 아나요?**

- [ ] **NumPy로 구현할 수 있나요?**

---

## 🎓 핵심 요약

1. **Attention**: 문맥을 파악하는 메커니즘
2. **공식**: softmax(QK^T / √d_k) V
3. **Q**: 질문, **K**: 키, **V**: 값
4. **Scaling**: 수치 안정성
5. **Self-Attention**: Q=K=V (같은 문장 내)

### 다음 학습
- **Day 44**: Multi-Head Attention
  - 여러 관점에서 동시에!

---

**수고하셨습니다!** 🎉

**Attention은 Transformer의 핵심입니다!**
