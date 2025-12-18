# Day 45-46: Transformer 아키텍처 (2시간) ⭐

## 📚 학습 목표
- **Transformer의 전체 구조 완벽히 이해하기**
- Encoder-Decoder 아키텍처 파악하기
- Positional Encoding의 원리 이해하기
- Layer Normalization과 Residual Connection 이해하기
- 전체 흐름 구현하기

---

## 🎯 강의 주제
**"LLM의 기본 설계도"**

---

## 📖 핵심 개념

### 1. Transformer 전체 구조

```
입력 토큰
    ↓
Embedding + Positional Encoding
    ↓
┌─────────────────────────────┐
│    Encoder (N개 층)          │
│  ┌─────────────────────┐   │
│  │ Multi-Head Attention │   │
│  │         ↓            │   │
│  │  Add & Norm          │   │
│  │         ↓            │   │
│  │  Feed Forward        │   │
│  │         ↓            │   │
│  │  Add & Norm          │   │
│  └─────────────────────┘   │
│  (N번 반복)                 │
└─────────────────────────────┘
    ↓
Encoder 출력
    ↓
┌─────────────────────────────┐
│    Decoder (N개 층)          │
│  ┌─────────────────────┐   │
│  │ Masked Self-Attn    │   │
│  │         ↓            │   │
│  │  Add & Norm          │   │
│  │         ↓            │   │
│  │ Cross-Attention      │   │
│  │  (Encoder 참조)      │   │
│  │         ↓            │   │
│  │  Add & Norm          │   │
│  │         ↓            │   │
│  │  Feed Forward        │   │
│  │         ↓            │   │
│  │  Add & Norm          │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
    ↓
Linear + Softmax
    ↓
출력 토큰 확률
```

---

### 2. Positional Encoding

**문제**:
```
Attention은 순서를 모름!
"고양이가 쥐를" = "쥐를 고양이가" (순서 정보 없음)
```

**해결**:
```
위치 정보를 임베딩에 추가!
```

**공식** (Sinusoidal):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos: 위치 (0, 1, 2, ...)
i: 차원 인덱스
d_model: 모델 차원
```

**특징**:
```
- 각 위치마다 고유한 패턴
- 상대적 위치 계산 가능
- 학습 불필요 (고정)
```

---

### 3. Feed Forward Network (FFN)

**구조**:
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

W₁: (d_model, d_ff)  - 확장
W₂: (d_ff, d_model)  - 축소

d_ff ≈ 4 × d_model  (일반적)
```

**역할**:
```
Attention: 토큰 간 상호작용
FFN: 각 토큰의 표현을 독립적으로 변환
```

---

### 4. Layer Normalization

**공식**:
```
LayerNorm(x) = γ × (x - μ) / σ + β

μ: 평균
σ: 표준편차
γ, β: 학습 가능한 파라미터
```

**효과**:
```
- 학습 안정화
- 수렴 속도 향상
- 기울기 소실/폭주 방지
```

---

### 5. Residual Connection

**공식**:
```
output = LayerNorm(x + Sublayer(x))

x: 입력
Sublayer: Attention 또는 FFN
```

**효과**:
```
- 깊은 네트워크에서 기울기 전달
- 항등 함수 학습 가능
- 학습 안정성
```

---

## 💻 Python 실습

### 실습 1: Positional Encoding
```python
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    """Sinusoidal Positional Encoding"""
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Sin for even indices
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))

            # Cos for odd indices
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return PE

# 생성
seq_len = 100
d_model = 128

PE = get_positional_encoding(seq_len, d_model)

print("=== Positional Encoding ===\n")
print(f"형태: {PE.shape}")
print(f"범위: [{PE.min():.2f}, {PE.max():.2f}]\n")

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 히트맵
im = ax1.imshow(PE.T, cmap='RdBu', aspect='auto')
ax1.set_xlabel('Position')
ax1.set_ylabel('Dimension')
ax1.set_title('Positional Encoding Heatmap')
plt.colorbar(im, ax=ax1)

# 특정 위치들
positions = [0, 10, 50, 99]
for pos in positions:
    ax2.plot(PE[pos, :64], label=f'pos={pos}', alpha=0.7)

ax2.set_xlabel('Dimension')
ax2.set_ylabel('Value')
ax2.set_title('Positional Encoding Patterns (first 64 dims)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=150)
print("Positional Encoding 시각화 저장!")
```

### 실습 2: Layer Normalization
```python
import numpy as np

class LayerNorm:
    """Layer Normalization"""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)  # 학습 가능
        self.beta = np.zeros(d_model)  # 학습 가능
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) 또는 (seq_len, d_model)

        Returns:
            normalized: same shape as x
        """
        # 마지막 차원에 대해 정규화
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)

        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta

        return output

# 테스트
print("\n=== Layer Normalization ===\n")

x = np.random.randn(2, 4, 8)  # (batch=2, seq_len=4, d_model=8)
ln = LayerNorm(d_model=8)

output = ln.forward(x)

print(f"입력 형태: {x.shape}")
print(f"출력 형태: {output.shape}\n")

# 정규화 확인
print("정규화 확인 (각 토큰):")
for i in range(2):
    for j in range(4):
        mean = output[i, j].mean()
        std = output[i, j].std()
        print(f"  배치 {i}, 토큰 {j}: mean={mean:.6f}, std={std:.6f}")

print("\n→ 평균 ≈ 0, 표준편차 ≈ 1")
```

### 실습 3: Transformer Encoder Layer
```python
import numpy as np

# 이전에 정의한 함수들 재사용
# (scaled_dot_product_attention, MultiHeadAttention, LayerNorm)

class FeedForward:
    """Position-wise Feed Forward Network"""

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """FFN(x) = ReLU(xW1 + b1)W2 + b2"""
        # 첫 번째 선형 + ReLU
        hidden = x @ self.W1 + self.b1
        hidden = np.maximum(0, hidden)  # ReLU

        # 두 번째 선형
        output = hidden @ self.W2 + self.b2

        return output


class TransformerEncoderLayer:
    """Transformer Encoder Layer"""

    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (seq_len, d_model)

        Returns:
            output: (seq_len, d_model)
        """
        # 1. Multi-Head Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)  # Residual

        # 2. Feed Forward + Residual + Norm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)  # Residual

        return x


# 사용 예
print("\n=== Transformer Encoder Layer ===\n")

seq_len = 5
d_model = 16
num_heads = 4
d_ff = 64

# 입력
x = np.random.randn(seq_len, d_model)

# Encoder Layer
encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
output = encoder_layer.forward(x)

print(f"입력: {x.shape}")
print(f"출력: {output.shape}")
print("\n→ 형태 유지 (seq_len, d_model)")
```

### 실습 4: 전체 Transformer (간소화)
```python
class SimpleTransformer:
    """간소화된 Transformer (Encoder만)"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        self.d_model = d_model

        # Embedding
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1

        # Positional Encoding
        max_len = 1000
        self.pos_encoding = get_positional_encoding(max_len, d_model)

        # Encoder Layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Output layer
        self.output_layer = np.random.randn(d_model, vocab_size) * 0.1

    def forward(self, token_ids):
        """
        Args:
            token_ids: list of token IDs

        Returns:
            logits: (seq_len, vocab_size)
        """
        seq_len = len(token_ids)

        # 1. Embedding
        x = self.embedding[token_ids]  # (seq_len, d_model)

        # 2. Positional Encoding
        x = x + self.pos_encoding[:seq_len, :]

        # 3. Encoder Layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x)

        # 4. Output projection
        logits = x @ self.output_layer  # (seq_len, vocab_size)

        return logits


# 사용 예
print("\n=== Simple Transformer ===\n")

vocab_size = 100
d_model = 64
num_heads = 8
d_ff = 256
num_layers = 3

transformer = SimpleTransformer(vocab_size, d_model, num_heads, d_ff, num_layers)

# 입력 토큰
token_ids = [5, 23, 67, 12]  # "The cat sat ."

# Forward
logits = transformer.forward(token_ids)

print(f"입력 토큰: {token_ids}")
print(f"출력 로짓: {logits.shape}")
print(f"  → 각 위치에서 다음 토큰에 대한 {vocab_size}개 점수")

# Softmax로 확률
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
print(f"\n확률 분포: {probs.shape}")
print(f"  → 각 위치에서 다음 토큰 확률")

# 가장 가능성 높은 토큰
next_tokens = probs.argmax(axis=-1)
print(f"\n예측된 다음 토큰: {next_tokens}")
```

---

## ✍️ 주요 컴포넌트 정리

### Transformer의 핵심 구성요소

| 컴포넌트 | 역할 | 수식 |
|----------|------|------|
| **Embedding** | 토큰 → 벡터 | E[token_id] |
| **Positional Encoding** | 위치 정보 추가 | sin/cos 함수 |
| **Multi-Head Attention** | 문맥 파악 | softmax(QK^T/√d_k)V |
| **Feed Forward** | 표현 변환 | ReLU(xW₁)W₂ |
| **Layer Norm** | 정규화 | (x-μ)/σ × γ + β |
| **Residual** | 기울기 전달 | x + Sublayer(x) |

---

## 🔗 LLM 연결점

### 1. GPT (Decoder-only)
```
Transformer Decoder만 사용
- Masked Self-Attention
- No Encoder
- Autoregressive 생성
```

### 2. BERT (Encoder-only)
```
Transformer Encoder만 사용
- Bidirectional
- Masked Language Modeling
- 양방향 문맥 활용
```

### 3. T5 (Full Transformer)
```
Encoder + Decoder
- Text-to-Text
- 모든 NLP 태스크 통합
```

---

## ✅ 체크포인트

- [ ] **Transformer의 전체 구조를 그릴 수 있나요?**

- [ ] **Positional Encoding의 필요성을 이해했나요?**

- [ ] **Residual Connection과 Layer Norm의 역할을 아나요?**

- [ ] **Encoder와 Decoder의 차이를 설명할 수 있나요?**

- [ ] **전체 흐름을 구현할 수 있나요?**

---

## 🎓 핵심 요약

1. **구조**: Encoder-Decoder 또는 한쪽만
2. **Attention**: Multi-Head Self/Cross Attention
3. **위치**: Positional Encoding
4. **안정성**: Layer Norm + Residual
5. **표현**: Feed Forward Network

### 다음 학습
- **Day 47**: LLM 학습의 완전한 수학
  - 모든 개념 통합!

---

**수고하셨습니다!** 🎉

**Transformer는 현대 LLM의 기초입니다!**
