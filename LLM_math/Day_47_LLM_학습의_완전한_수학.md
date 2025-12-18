# Day 47: LLM 학습의 완전한 수학 (2시간) ⭐

## 📚 학습 목표
- **LLM 학습의 전체 흐름 완벽히 이해하기**
- 모든 수학 개념의 통합 파악하기
- Forward와 Backward의 전체 과정 이해하기
- 실제 학습 코드 구현하기

---

## 🎯 강의 주제
**"모든 수학이 하나로 - LLM 학습"**

---

## 📖 전체 흐름

### 1. Forward Pass (예측)

```
입력 토큰: [1, 5, 23, 67]
    ↓
1. Embedding
   토큰 → 벡터 (선형대수)
   E[token_id] ∈ ℝ^d_model

2. Positional Encoding
   위치 정보 추가
   PE = sin/cos (삼각함수)

3. Transformer Layers (N개)
   각 층:
   a. Multi-Head Attention
      - Q, K, V 투영 (행렬곱)
      - QK^T (내적, 유사도)
      - / √d_k (정규화)
      - Softmax (확률, 지수/로그)
      - × V (가중합)

   b. Feed Forward
      - ReLU(xW₁)W₂ (미분 가능 활성화)

   c. Layer Norm + Residual
      - 정규화, 기울기 전달

4. Output Layer
   logits = hidden @ W_out
   P(token|context) = softmax(logits)

5. Loss Calculation
   L = CrossEntropy(실제, 예측)
   L = -log P(실제_토큰|context)
   → 정보이론!
```

---

### 2. Backward Pass (학습)

```
손실 L에서 시작
    ↓
1. Output Layer
   dL/dW_out (편미분)

2. Transformer Layers (역순)
   각 층:
   a. Layer Norm + Residual
      연쇄법칙 적용

   b. Feed Forward
      dL/dW₂, dL/dW₁
      ReLU 미분 (x>0: 1, x<0: 0)

   c. Multi-Head Attention
      dL/dW_O, dL/dW_V, dL/dW_K, dL/dW_Q
      Softmax 미분, 행렬 전치

3. Embedding
   dL/dE (각 토큰 임베딩 업데이트)

4. Gradient Descent
   θ_new = θ_old - α × dL/dθ
   → 최적화!
```

---

### 3. 사용된 모든 수학

**기초 (Day 1-10)**:
```
- 벡터, 행렬
- 지수, 로그
- 함수, 그래프
```

**선형대수 (Day 11-20)**:
```
- 내적: QK^T
- 정규화: v/||v||
- 행렬곱: 모든 선형 층
- 전치: K^T
```

**미적분 (Day 21-27)**:
```
- 미분: 기울기 계산
- 연쇄법칙: Backpropagation
- 편미분: 파라미터별 기울기
- 경사하강법: 최적화
```

**확률과 정보이론 (Day 28-38)**:
```
- 확률: Softmax 출력
- Cross Entropy: 손실 함수
- Perplexity: 모델 평가
- 정보 이득: BPE
```

**LLM 핵심 (Day 39-46)**:
```
- BPE: 토큰화
- Attention: 문맥 파악
- Transformer: 전체 아키텍처
```

---

## 💻 완전한 학습 루프 구현

### 전체 학습 코드
```python
import numpy as np

class TinyLM:
    """완전한 언어 모델 (교육용)"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

        # Positional Encoding
        self.pos_encoding = self._get_positional_encoding(1000, d_model)

        # Transformer Layers (간소화)
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'W_Q': np.random.randn(d_model, d_model) * 0.01,
                'W_K': np.random.randn(d_model, d_model) * 0.01,
                'W_V': np.random.randn(d_model, d_model) * 0.01,
                'W_O': np.random.randn(d_model, d_model) * 0.01,
                'W1': np.random.randn(d_model, d_ff) * 0.01,
                'W2': np.random.randn(d_ff, d_model) * 0.01,
            }
            self.layers.append(layer)

        # Output
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01

    def _get_positional_encoding(self, max_len, d_model):
        PE = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        return PE

    def forward(self, token_ids, store_activations=False):
        """Forward pass"""
        seq_len = len(token_ids)

        # 1. Embedding + Positional
        x = self.embedding[token_ids] + self.pos_encoding[:seq_len]

        activations = {'input': x} if store_activations else None

        # 2. Transformer Layers (간소화된 버전)
        for i, layer in enumerate(self.layers):
            # Self-Attention (간소화)
            Q = x @ layer['W_Q']
            K = x @ layer['W_K']
            V = x @ layer['W_V']

            scores = (Q @ K.T) / np.sqrt(self.d_model)
            attn = self._softmax(scores)
            attn_out = attn @ V
            attn_out = attn_out @ layer['W_O']

            # Residual (Layer Norm 생략)
            x = x + attn_out

            # FFN
            ffn_out = np.maximum(0, x @ layer['W1']) @ layer['W2']
            x = x + ffn_out

            if store_activations:
                activations[f'layer_{i}'] = x

        # 3. Output
        logits = x @ self.W_out

        return logits, activations

    def _softmax(self, x):
        """Softmax (수치 안정성)"""
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def compute_loss(self, token_ids, target_ids):
        """Cross Entropy Loss"""
        logits, _ = self.forward(token_ids)

        # Softmax
        probs = self._softmax(logits)

        # Cross Entropy
        loss = 0
        for i, target in enumerate(target_ids):
            loss += -np.log(probs[i, target] + 1e-10)

        return loss / len(target_ids)

    def train_step(self, token_ids, target_ids, learning_rate):
        """한 스텝 학습 (간소화)"""
        # Forward
        loss_before = self.compute_loss(token_ids, target_ids)

        # Backward (수치 미분으로 근사)
        # 실제로는 역전파 사용
        epsilon = 1e-5

        # Embedding 업데이트 (예시)
        for idx in token_ids:
            grad = np.zeros_like(self.embedding[idx])

            for j in range(self.d_model):
                self.embedding[idx, j] += epsilon
                loss_plus = self.compute_loss(token_ids, target_ids)
                self.embedding[idx, j] -= epsilon

                grad[j] = (loss_plus - loss_before) / epsilon

            self.embedding[idx] -= learning_rate * grad

        # Forward again
        loss_after = self.compute_loss(token_ids, target_ids)

        return loss_before, loss_after


# 사용 예
print("=== Tiny Language Model 학습 ===\n")

# 초기화
vocab_size = 50
d_model = 32
num_heads = 4
d_ff = 64
num_layers = 2

model = TinyLM(vocab_size, d_model, num_heads, d_ff, num_layers)

# 학습 데이터
# "The cat sat" → 다음 토큰 예측
sequences = [
    ([5, 12, 23], [12, 23, 7]),   # "The cat sat" → "cat sat ."
    ([5, 8, 15], [8, 15, 7]),      # ...
]

print(f"모델 초기화 완료")
print(f"- Vocab: {vocab_size}")
print(f"- d_model: {d_model}")
print(f"- Layers: {num_layers}\n")

# 학습
epochs = 5
learning_rate = 0.001

print("학습 시작...\n")

for epoch in range(epochs):
    total_loss = 0

    for token_ids, target_ids in sequences:
        loss, _ = model.train_step(token_ids, target_ids, learning_rate)
        total_loss += loss

    avg_loss = total_loss / len(sequences)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

print("\n학습 완료!")

# 예측
print("\n=== 예측 ===\n")
test_input = [5, 12]  # "The cat"
logits, _ = model.forward(test_input)

# 마지막 토큰의 예측
last_probs = model._softmax(logits[-1])
predicted_token = last_probs.argmax()

print(f"입력: {test_input}")
print(f"예측된 다음 토큰: {predicted_token}")
print(f"확률: {last_probs[predicted_token]:.4f}")
```

---

## 📊 수학 개념 통합 맵

```
┌─────────────────────────────────────────┐
│          LLM 학습 과정                   │
└─────────────────────────────────────────┘
                ↓
┌─────────────┬─────────────┬──────────────┐
│  Forward    │  Loss       │  Backward    │
└─────────────┴─────────────┴──────────────┘
      ↓             ↓              ↓
  선형대수      정보이론        미적분
  - 내적        - 엔트로피     - 편미분
  - 행렬곱      - Cross Ent   - 연쇄법칙
  - 정규화      - Perplexity  - 경사하강법
```

---

## 🔗 실제 LLM과의 비교

### GPT-3
```
- 175B 파라미터
- 96 layers
- d_model = 12288
- num_heads = 96
- Vocab = 50257 (BPE)

학습:
- 배치 크기: 3.2M 토큰
- Adam 최적화
- Learning rate scheduling
- Gradient clipping
```

### 우리의 Tiny LM
```
- ~10K 파라미터
- 2 layers
- d_model = 32
- num_heads = 4

→ 같은 원리, 작은 규모!
```

---

## ✅ 최종 체크포인트

- [ ] **Forward pass의 모든 단계를 설명할 수 있나요?**

- [ ] **Backward pass와 연쇄법칙을 이해했나요?**

- [ ] **각 단계에서 사용된 수학을 말할 수 있나요?**

- [ ] **손실 함수에서 파라미터 업데이트까지 흐름을 아나요?**

- [ ] **실제 LLM과의 차이를 이해했나요?**

---

## 🎓 핵심 요약

**LLM 학습 = 모든 수학의 통합**

1. **입력**: 토큰 → 임베딩 (선형대수)
2. **처리**: Attention + FFN (행렬, 미분)
3. **출력**: Softmax (확률)
4. **손실**: Cross Entropy (정보이론)
5. **학습**: Backprop + GD (미적분)

**당신은 이제 LLM의 수학을 완전히 이해했습니다!**

### 다음 학습
- **Day 48**: 최종 프로젝트
  - Tiny Language Model 구현!

---

**축하합니다!** 🎉

**모든 수학이 하나로 연결되었습니다!**
