# Day 44: Multi-Head Attention (1.5시간) ⭐

## 📚 학습 목표
- **Multi-Head Attention의 원리 완벽히 이해하기**
- 여러 헤드가 왜 필요한지 파악하기
- Parallel Attention 계산 이해하기
- 전체 흐름 구현하기

---

## 🎯 강의 주제
**"여러 관점에서 동시에 보기"**

---

## 📖 핵심 개념

### 1. 왜 Multi-Head인가?

**문제**:
```
하나의 Attention:
- 한 가지 패턴만 학습
- "고양이가 쥐를 잡았다"
  → "고양이 - 잡았다" (주어-서술어)만 포착

다양한 관계를 동시에 파악하려면?
```

**해결책**:
```
여러 개의 Attention을 병렬로 실행!

Head 1: 주어-서술어 관계
Head 2: 목적어-동사 관계
Head 3: 형용사-명사 관계
...
```

---

### 2. Multi-Head Attention 공식

**전체 과정**:
```
1. 입력을 h개의 헤드로 분할
2. 각 헤드에서 독립적으로 Attention
3. 결과를 연결(concat)
4. 최종 선형 변환
```

**수식**:
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O

head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

W^Q_i, W^K_i, W^V_i: 각 헤드의 투영 행렬
W^O: 출력 투영 행렬
```

---

### 3. 차원 관리

**핵심 아이디어**:
```
전체 모델 차원: d_model = 512
헤드 수: h = 8

각 헤드의 차원: d_k = d_v = d_model / h = 64

→ 계산량은 거의 동일하면서 다양한 표현 학습!
```

**차원 변화**:
```
입력: (batch, seq_len, d_model)

1. 투영: (batch, seq_len, d_model) → (batch, seq_len, d_k) × h
2. Reshape: (batch, h, seq_len, d_k)
3. Attention: 각 헤드에서 독립적으로
4. Concat: (batch, seq_len, h × d_k) = (batch, seq_len, d_model)
5. 출력 투영: (batch, seq_len, d_model)
```

---

### 4. 예제: 2-Head Attention

**설정**:
```
seq_len = 3 (단어 3개)
d_model = 4
h = 2 (헤드 2개)
d_k = d_model / h = 2
```

**입력**:
```
X = [[1, 2, 3, 4],    # 단어 1
     [5, 6, 7, 8],    # 단어 2
     [9, 10, 11, 12]] # 단어 3

(3, 4)
```

**Head 1**:
```
W^Q_1: (4, 2) - 처음 2차원 투영
Q_1 = X @ W^Q_1 → (3, 2)

K_1 = X @ W^K_1 → (3, 2)
V_1 = X @ W^V_1 → (3, 2)

output_1 = Attention(Q_1, K_1, V_1) → (3, 2)
```

**Head 2**:
```
W^Q_2: (4, 2) - 다른 2차원 투영
Q_2 = X @ W^Q_2 → (3, 2)

K_2 = X @ W^K_2 → (3, 2)
V_2 = X @ W^V_2 → (3, 2)

output_2 = Attention(Q_2, K_2, V_2) → (3, 2)
```

**Concat**:
```
output = [output_1 | output_2] → (3, 4)
```

**최종**:
```
final_output = output @ W^O → (3, 4)
```

---

## 💻 Python 실습

### 실습 1: Multi-Head Attention 구현
```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention"""
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    if mask is not None:
        scores += (mask * -1e9)

    attn_weights = np.exp(scores)
    attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

    output = attn_weights @ V
    return output, attn_weights


class MultiHeadAttention:
    """Multi-Head Attention"""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 가중치 초기화
        self.W_Q = np.random.randn(num_heads, d_model, self.d_k) * 0.1
        self.W_K = np.random.randn(num_heads, d_model, self.d_k) * 0.1
        self.W_V = np.random.randn(num_heads, d_model, self.d_k) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (seq_len, d_model)
            mask: optional

        Returns:
            output: (seq_len, d_model)
            attn_weights: list of (seq_len, seq_len) for each head
        """
        seq_len = Q.shape[0]

        # 각 헤드별 출력 저장
        head_outputs = []
        all_attn_weights = []

        for i in range(self.num_heads):
            # 투영
            Q_i = Q @ self.W_Q[i]  # (seq_len, d_k)
            K_i = K @ self.W_K[i]  # (seq_len, d_k)
            V_i = V @ self.W_V[i]  # (seq_len, d_k)

            # Attention
            head_output, attn_weights = scaled_dot_product_attention(
                Q_i, K_i, V_i, mask
            )

            head_outputs.append(head_output)
            all_attn_weights.append(attn_weights)

        # Concat
        concat_output = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)

        # 최종 투영
        output = concat_output @ self.W_O  # (seq_len, d_model)

        return output, all_attn_weights


# 사용 예
print("=== Multi-Head Attention ===\n")

# 설정
seq_len = 4
d_model = 8
num_heads = 2

# 입력
X = np.random.randn(seq_len, d_model)

print(f"입력 형태: {X.shape}")
print(f"d_model: {d_model}")
print(f"num_heads: {num_heads}")
print(f"d_k (각 헤드): {d_model // num_heads}\n")

# Multi-Head Attention
mha = MultiHeadAttention(d_model, num_heads)
output, attn_weights = mha.forward(X, X, X)

print(f"출력 형태: {output.shape}")
print(f"Attention 헤드 수: {len(attn_weights)}")
print(f"각 헤드 attention 형태: {attn_weights[0].shape}\n")

# 각 헤드의 attention 확인
for i, attn in enumerate(attn_weights):
    print(f"Head {i+1} attention weights:")
    print(attn)
    print()
```

### 실습 2: 헤드별 Attention 패턴 시각화
```python
import numpy as np
import matplotlib.pyplot as plt

# 문장
words = ["The", "cat", "sat", "mat"]
seq_len = len(words)
d_model = 8
num_heads = 4

# 입력
np.random.seed(42)
X = np.random.randn(seq_len, d_model)

# Multi-Head Attention
mha = MultiHeadAttention(d_model, num_heads)
output, attn_weights = mha.forward(X, X, X)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, attn in enumerate(attn_weights):
    ax = axes[i]

    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(words)
    ax.set_yticklabels(words)
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    ax.set_title(f'Head {i+1} Attention')

    # 값 표시
    for row in range(seq_len):
        for col in range(seq_len):
            text = ax.text(col, row, f'{attn[row, col]:.2f}',
                          ha="center", va="center",
                          color="black", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('multi_head_attention.png', dpi=150)
print("Multi-Head Attention 시각화 저장 완료!")
print("\n→ 각 헤드가 다른 패턴을 학습함!")
```

### 실습 3: 헤드 수에 따른 비교
```python
import numpy as np

def test_num_heads(seq_len, d_model, num_heads_list):
    """다양한 헤드 수 테스트"""
    print("=== 헤드 수 비교 ===\n")

    X = np.random.randn(seq_len, d_model)

    for num_heads in num_heads_list:
        if d_model % num_heads != 0:
            print(f"num_heads={num_heads}: 불가능 (d_model % num_heads != 0)")
            continue

        mha = MultiHeadAttention(d_model, num_heads)
        output, _ = mha.forward(X, X, X)

        d_k = d_model // num_heads
        total_params = (
            num_heads * d_model * d_k * 3 +  # W_Q, W_K, W_V
            d_model * d_model                  # W_O
        )

        print(f"num_heads={num_heads}:")
        print(f"  d_k (각 헤드 차원): {d_k}")
        print(f"  파라미터 수: {total_params}")
        print(f"  출력 형태: {output.shape}")
        print()

# 테스트
test_num_heads(seq_len=4, d_model=64, num_heads_list=[1, 2, 4, 8, 16])
```

### 실습 4: Self vs Cross Attention
```python
import numpy as np

# Self-Attention: Q=K=V (같은 시퀀스)
print("=== Self-Attention vs Cross-Attention ===\n")

# Encoder 출력 (다른 문장)
encoder_output = np.random.randn(5, 8)  # 5 단어

# Decoder 입력 (생성 중인 문장)
decoder_input = np.random.randn(3, 8)   # 3 단어

mha = MultiHeadAttention(d_model=8, num_heads=2)

# 1. Self-Attention (Decoder)
print("1. Decoder Self-Attention:")
self_output, self_attn = mha.forward(decoder_input, decoder_input, decoder_input)
print(f"   Query=Key=Value: {decoder_input.shape}")
print(f"   출력: {self_output.shape}")
print(f"   Attention: {self_attn[0].shape} (3x3)\n")

# 2. Cross-Attention (Encoder-Decoder)
print("2. Encoder-Decoder Cross-Attention:")
cross_output, cross_attn = mha.forward(decoder_input, encoder_output, encoder_output)
print(f"   Query (Decoder): {decoder_input.shape}")
print(f"   Key,Value (Encoder): {encoder_output.shape}")
print(f"   출력: {cross_output.shape}")
print(f"   Attention: {cross_attn[0].shape} (3x5)")
print("   → Decoder가 Encoder의 모든 위치를 볼 수 있음!")
```

---

## ✍️ 손 계산 연습

### 연습: 2-Head Attention (간소화)
```
입력: X = [1, 2]  (1개 단어, d_model=2)

h = 2, d_k = 1

Head 1:
  W^Q_1 = [0.5], W^K_1 = [0.5], W^V_1 = [0.5]
  Q_1 = [1,2] @ [0.5] = 1.5
  K_1 = 1.5, V_1 = 1.5
  output_1 = Attention(1.5, 1.5, 1.5) = 1.5

Head 2:
  W^Q_2 = [0.3], W^K_2 = [0.3], W^V_2 = [0.3]
  Q_2 = [1,2] @ [0.3] = 0.9
  K_2 = 0.9, V_2 = 0.9
  output_2 = 0.9

Concat: [1.5, 0.9]

최종: [1.5, 0.9] @ W^O
```

---

## 🔗 LLM 연결점

### 1. GPT-3 구성
```
- 96 layers
- d_model = 12288
- num_heads = 96
- d_k = 128

→ 각 층에서 96개의 다른 관점!
```

### 2. BERT
```
- 12 layers (Base) / 24 (Large)
- d_model = 768 (Base) / 1024 (Large)
- num_heads = 12 / 16
```

---

## ✅ 체크포인트

- [ ] **Multi-Head가 왜 필요한지 이해했나요?**

- [ ] **차원 관리 (d_model / h = d_k)를 이해했나요?**

- [ ] **Self vs Cross Attention의 차이를 아나요?**

- [ ] **구현할 수 있나요?**

---

## 🎓 핵심 요약

1. **Multi-Head**: 여러 관점 병렬 학습
2. **차원 분할**: d_k = d_model / h
3. **과정**: 투영 → Attention × h → Concat → 투영
4. **효과**: 다양한 패턴 포착, 표현력 향상

### 다음 학습
- **Day 45-46**: Transformer 아키텍처
  - 전체 구조 완성!

---

**수고하셨습니다!** 🎉

**Multi-Head Attention은 Transformer의 핵심 혁신입니다!**
