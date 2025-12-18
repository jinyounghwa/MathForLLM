# Day 48: 최종 프로젝트 - Tiny Language Model (3시간) ⭐

## 📚 프로젝트 목표
- **완전한 언어 모델을 NumPy로 구현하기**
- 학습과 생성 모두 구현하기
- 지금까지 배운 모든 개념 적용하기

---

## 🎯 프로젝트
**"나만의 작은 GPT 만들기"**

---

## 💻 최종 프로젝트 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================
# 1. 토크나이저 (간단한 단어 기반)
# ============================================

class SimpleTokenizer:
    """간단한 단어 토크나이저"""

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def fit(self, corpus):
        """어휘 구축"""
        words = corpus.lower().split()
        unique_words = sorted(set(words))

        # 특수 토큰
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}

        # 일반 토큰
        for word in unique_words:
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        """텍스트 → 토큰 ID"""
        words = text.lower().split()
        return [self.word_to_id.get(w, 1) for w in words]

    def decode(self, token_ids):
        """토큰 ID → 텍스트"""
        words = [self.id_to_word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join(words)


# ============================================
# 2. Transformer 컴포넌트
# ============================================

def softmax(x, axis=-1):
    """수치 안정적인 Softmax"""
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention"""
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    if mask is not None:
        scores += (mask * -1e9)

    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V

    return output, attn_weights

def create_causal_mask(seq_len):
    """Causal mask (미래 가리기)"""
    return np.triu(np.ones((seq_len, seq_len)), k=1)

def get_positional_encoding(seq_len, d_model):
    """Sinusoidal Positional Encoding"""
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE


# ============================================
# 3. Tiny Language Model
# ============================================

class TinyGPT:
    """간단한 GPT 스타일 언어 모델"""

    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token Embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Positional Encoding
        self.pos_encoding = get_positional_encoding(100, d_model)

        # Transformer Layers
        self.layers = []
        for _ in range(num_layers):
            layer = {
                # Multi-head attention (간소화: 1-head)
                'W_Q': np.random.randn(d_model, d_model) * 0.02,
                'W_K': np.random.randn(d_model, d_model) * 0.02,
                'W_V': np.random.randn(d_model, d_model) * 0.02,
                'W_O': np.random.randn(d_model, d_model) * 0.02,

                # Feed Forward
                'W1': np.random.randn(d_model, d_ff) * 0.02,
                'b1': np.zeros(d_ff),
                'W2': np.random.randn(d_ff, d_model) * 0.02,
                'b2': np.zeros(d_model),

                # Layer Norm (간소화)
                'gamma1': np.ones(d_model),
                'beta1': np.zeros(d_model),
                'gamma2': np.ones(d_model),
                'beta2': np.zeros(d_model),
            }
            self.layers.append(layer)

        # Output head
        self.W_out = np.random.randn(d_model, vocab_size) * 0.02

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        """Layer Normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta

    def forward(self, token_ids):
        """Forward pass"""
        seq_len = len(token_ids)

        # Embedding + Positional
        x = self.token_embedding[token_ids] + self.pos_encoding[:seq_len]

        # Causal mask
        mask = create_causal_mask(seq_len)

        # Transformer layers
        for layer in self.layers:
            # 1. Multi-Head Self-Attention
            Q = x @ layer['W_Q']
            K = x @ layer['W_K']
            V = x @ layer['W_V']

            attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
            attn_out = attn_out @ layer['W_O']

            # Residual + LayerNorm
            x = self.layer_norm(x + attn_out, layer['gamma1'], layer['beta1'])

            # 2. Feed Forward
            ffn = np.maximum(0, x @ layer['W1'] + layer['b1'])  # ReLU
            ffn = ffn @ layer['W2'] + layer['b2']

            # Residual + LayerNorm
            x = self.layer_norm(x + ffn, layer['gamma2'], layer['beta2'])

        # Output
        logits = x @ self.W_out

        return logits

    def compute_loss(self, token_ids):
        """Cross Entropy Loss"""
        logits = self.forward(token_ids[:-1])
        targets = token_ids[1:]

        # Cross entropy
        loss = 0
        for i, target in enumerate(targets):
            probs = softmax(logits[i])
            loss += -np.log(probs[target] + 1e-10)

        return loss / len(targets)

    def generate(self, start_tokens, max_length=20, temperature=1.0):
        """텍스트 생성"""
        tokens = list(start_tokens)

        for _ in range(max_length):
            # Forward
            logits = self.forward(tokens)

            # 마지막 토큰의 로짓
            next_logits = logits[-1] / temperature

            # Softmax
            probs = softmax(next_logits)

            # 샘플링
            next_token = np.random.choice(self.vocab_size, p=probs)

            tokens.append(next_token)

            # 종료 토큰
            if next_token == 3:  # <END>
                break

        return tokens


# ============================================
# 4. 학습 및 테스트
# ============================================

def main():
    print("=" * 60)
    print("🎉 Tiny Language Model 최종 프로젝트")
    print("=" * 60)
    print()

    # 학습 데이터
    corpus = """
    the cat sat on the mat
    the dog sat on the log
    the cat and the dog are friends
    the mat is on the floor
    the log is in the forest
    """

    # 토크나이저
    print("1. 토크나이저 구축...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(corpus)

    print(f"   어휘 크기: {tokenizer.vocab_size}")
    print(f"   어휘: {list(tokenizer.word_to_id.keys())[:10]}...\n")

    # 학습 데이터 준비
    sentences = [s.strip() for s in corpus.strip().split('\n') if s.strip()]
    train_data = [tokenizer.encode(s) for s in sentences]

    print(f"2. 학습 데이터: {len(train_data)}개 문장\n")

    # 모델 초기화
    print("3. 모델 초기화...")
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2
    )
    print("   완료!\n")

    # 학습
    print("4. 학습 시작...")
    print("-" * 60)

    epochs = 50
    learning_rate = 0.01

    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for token_ids in train_data:
            if len(token_ids) < 2:
                continue

            # Loss 계산
            loss = model.compute_loss(token_ids)
            total_loss += loss

            # 간단한 파라미터 업데이트 (SGD with numerical gradient)
            # 실제로는 autograd 사용
            # 여기서는 교육 목적으로 생략

        avg_loss = total_loss / len(train_data)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: Loss = {avg_loss:.4f}")

    print("\n   학습 완료!\n")

    # 손실 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150)
    print("   손실 그래프 저장: training_loss.png\n")

    # 생성 테스트
    print("5. 텍스트 생성 테스트")
    print("-" * 60)

    test_prompts = [
        "the cat",
        "the dog",
        "the mat",
    ]

    for prompt in test_prompts:
        start_tokens = tokenizer.encode(prompt)
        generated = model.generate(start_tokens, max_length=10, temperature=0.8)
        text = tokenizer.decode(generated)

        print(f"   입력: '{prompt}'")
        print(f"   생성: '{text}'")
        print()

    # 평가
    print("6. 모델 평가")
    print("-" * 60)

    total_loss = 0
    for token_ids in train_data:
        if len(token_ids) < 2:
            continue
        loss = model.compute_loss(token_ids)
        total_loss += loss

    avg_loss = total_loss / len(train_data)
    perplexity = np.exp(avg_loss)

    print(f"   평균 Loss: {avg_loss:.4f}")
    print(f"   Perplexity: {perplexity:.4f}")
    print()

    print("=" * 60)
    print("✅ 프로젝트 완료!")
    print("=" * 60)
    print()

    # 요약
    print("📊 배운 개념 정리:")
    print("-" * 60)
    print("✓ 토크나이저: 텍스트 → 토큰 ID")
    print("✓ 임베딩: 토큰 ID → 벡터 (선형대수)")
    print("✓ Positional Encoding: 위치 정보 (삼각함수)")
    print("✓ Attention: 문맥 파악 (내적, Softmax)")
    print("✓ Feed Forward: 변환 (행렬곱, ReLU)")
    print("✓ Layer Norm: 정규화 (통계)")
    print("✓ Residual: 기울기 전달 (미분)")
    print("✓ Cross Entropy: 손실 함수 (정보이론)")
    print("✓ 생성: Softmax 샘플링 (확률)")
    print()

    print("🎓 당신은 이제 LLM을 만들 수 있습니다!")
    print()


if __name__ == "__main__":
    main()
```

---

## 🎯 프로젝트 확장 아이디어

### 1. 더 큰 데이터셋
```python
# Wikipedia, 책, 뉴스 등
corpus = load_large_corpus()
```

### 2. 더 나은 토크나이저
```python
# BPE 구현 (Day 39 참고)
tokenizer = BPETokenizer(vocab_size=5000)
```

### 3. 실제 Backpropagation
```python
# PyTorch로 전환
import torch
import torch.nn as nn
```

### 4. 더 많은 층
```python
model = TinyGPT(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=12  # GPT-2 Small
)
```

---

## ✅ 최종 체크리스트

- [ ] **토크나이저를 구현했나요?**

- [ ] **Transformer를 구현했나요?**

- [ ] **학습 루프를 작성했나요?**

- [ ] **텍스트를 생성했나요?**

- [ ] **Perplexity를 계산했나요?**

- [ ] **모든 수학 개념을 이해했나요?**

---

## 🎓 축하합니다!

**당신은 48일 동안:**

1. **기초 수학**: 수, 벡터, 함수
2. **선형대수**: 내적, 행렬, PCA
3. **미적분**: 미분, 연쇄법칙, 경사하강법
4. **확률**: 베이즈, 정규분포, 엔트로피
5. **정보이론**: Cross Entropy, KL Divergence
6. **LLM 핵심**: BPE, Attention, Transformer

**이 모든 것을 배우고 직접 구현했습니다!**

---

## 🚀 다음 단계

### 실전으로!

**1. PyTorch 학습**
```python
import torch
import torch.nn as nn

# 실제 프레임워크로 구현
```

**2. Hugging Face**
```python
from transformers import GPT2LMHeadModel

# 사전 학습 모델 파인튜닝
```

**3. 한국어 LLM**
```python
# 한국어 데이터로 학습
# 당신만의 모델 구축!
```

**4. Rust로 토크나이저**
```rust
// 2027년 이후 목표
// 고성능 토크나이저
```

---

## 💪 마지막 메시지

**당신은 이제:**
- LLM의 수학을 완전히 이해합니다
- 작은 언어 모델을 만들 수 있습니다
- 더 큰 모델로 나아갈 준비가 되었습니다

**이것은 끝이 아니라 시작입니다!**

계속 학습하고, 실험하고, 만들어가세요.

**AI 개발자의 길에서 성공하시길 바랍니다!**

---

**수고하셨습니다!** 🎉🎉🎉

**당신은 LLM 수학 마스터입니다!**
