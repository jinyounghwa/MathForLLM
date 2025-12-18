# Day 4: 함수와 그래프 (1시간)

## 📚 학습 목표
- 함수의 본질을 "입력 → 계산 → 출력" 구조로 이해하기
- 주요 함수들(선형, 지수, 로그)의 특징 파악하기
- 합성함수의 개념과 신경망과의 연결 이해하기

---

## 🎯 강의 주제
**"함수 = 입력을 받아서 출력을 내는 기계"**

---

## 📖 핵심 개념

### 1. 함수의 정의

#### 1.1 함수란?
**입력(x)을 받아 정해진 규칙에 따라 출력(y)을 내는 관계**

```
f(x) = y
```

**구성 요소**:
- **정의역 (Domain)**: 입력 가능한 값들의 집합
- **공역 (Codomain)**: 출력 가능한 값들의 집합
- **치역 (Range)**: 실제로 출력되는 값들의 집합

**예시**:
```python
def f(x):
    return 2 * x + 1

f(3) = 7   # 입력 3 → 출력 7
f(5) = 11  # 입력 5 → 출력 11
```

---

### 2. 주요 함수들

#### 2.1 선형 함수 (Linear Function)
```
f(x) = ax + b
```

- **a**: 기울기 (slope)
- **b**: y절편 (intercept)

**특징**:
- 직선 그래프
- 일정한 변화율
- 가장 단순한 함수

**예시**:
```
f(x) = 2x + 1
f(0) = 1
f(1) = 3
f(2) = 5
```

**LLM 연결**: 선형 변환 (Linear Layer)
```python
y = Wx + b  # 신경망의 기본 연산
```

---

#### 2.2 지수 함수 (Exponential Function)
```
f(x) = aˣ  (특히 eˣ)
```

**특징**:
- 빠르게 증가 (폭발적 성장)
- 항상 양수
- 미분해도 자기 자신 (eˣ의 경우)

**그래프 형태**:
```
  ↑
  |     *
  |    *
  |   *
  |  *
  | *
  |*_____________→
```

**LLM 연결**: Softmax
```python
softmax(x) = exp(x) / Σ exp(x)
```

---

#### 2.3 로그 함수 (Logarithmic Function)
```
f(x) = log(x)
```

**특징**:
- 천천히 증가
- x > 0에서만 정의
- 지수 함수의 역함수

**그래프 형태**:
```
  ↑
  |         *****
  |      ***
  |   **
  | *
  |*
  |_____________→
```

**LLM 연결**: Log-Softmax, Cross-Entropy
```python
loss = -log(predicted_prob)
```

---

#### 2.4 이차 함수 (Quadratic Function)
```
f(x) = ax² + bx + c
```

**특징**:
- 포물선 모양
- 최댓값 또는 최솟값 존재
- a > 0: 아래로 볼록, a < 0: 위로 볼록

**LLM 연결**: 손실 함수 (Loss Function)
```
L(θ) = (y - ŷ)²  # MSE Loss
```

---

#### 2.5 시그모이드 함수 (Sigmoid Function)
```
σ(x) = 1 / (1 + e⁻ˣ)
```

**특징**:
- S자 모양
- 출력 범위: (0, 1)
- 확률로 해석 가능

**그래프**:
```
  1 ↑     ________
    |    /
0.5 |   *
    |  /
  0 |_/___________→
      0
```

**LLM 연결**: 이진 분류, Gate 메커니즘 (LSTM)
```python
gate = sigmoid(Wx + b)
```

---

#### 2.6 ReLU 함수
```
ReLU(x) = max(0, x) = {x if x > 0, 0 if x ≤ 0}
```

**그래프**:
```
  ↑
  |    /
  |   /
  |  /
  | /
  |/_____________→
  0
```

**특징**:
- 단순하고 빠름
- 기울기 소실 문제 해결
- 현대 신경망의 표준 활성화 함수

**LLM 연결**: Transformer의 FFN (Feed-Forward Network)
```python
output = ReLU(Wx + b)
```

---

### 3. 합성함수 (Function Composition)

#### 3.1 정의
**함수를 차례로 적용**

```
(f ∘ g)(x) = f(g(x))
```

**단계**:
1. g(x) 계산
2. 그 결과를 f에 넣기

**예시**:
```
f(x) = x²
g(x) = x + 1

(f ∘ g)(3) = f(g(3))
           = f(4)
           = 16
```

---

#### 3.2 신경망 = 합성함수!

**1층 신경망**:
```
h = ReLU(W₁x + b₁)
y = W₂h + b₂
```

**합성함수로 표현**:
```
y = f₂(f₁(x))
```

**깊은 신경망 (Deep Neural Network)**:
```
y = f_n(f_{n-1}(...f₂(f₁(x))))
```

**LLM (Transformer)**:
```
x → Embedding → Layer1 → Layer2 → ... → Layer_N → Output
```

각 층이 함수이고, 전체가 거대한 합성함수!

---

## 💻 Python 실습

### 실습 1: 기본 함수 정의와 사용
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 선형 함수
def linear(x):
    return 2 * x + 1

# 2. 지수 함수
def exponential(x):
    return np.exp(x)

# 3. 로그 함수
def logarithm(x):
    return np.log(x)

# 4. 이차 함수
def quadratic(x):
    return x**2 - 4*x + 3

# 5. 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 6. ReLU
def relu(x):
    return np.maximum(0, x)

# 테스트
x_test = 2
print(f"x = {x_test}")
print(f"linear({x_test}) = {linear(x_test)}")
print(f"exponential({x_test}) = {exponential(x_test):.4f}")
print(f"logarithm({x_test}) = {logarithm(x_test):.4f}")
print(f"quadratic({x_test}) = {quadratic(x_test)}")
print(f"sigmoid({x_test}) = {sigmoid(x_test):.4f}")
print(f"relu({x_test}) = {relu(x_test)}")
```

### 실습 2: 함수 시각화
```python
import numpy as np
import matplotlib.pyplot as plt

# x 범위 설정
x_exp = np.linspace(-2, 2, 100)
x_log = np.linspace(0.1, 5, 100)
x_sigmoid = np.linspace(-6, 6, 100)
x_relu = np.linspace(-5, 5, 100)

# 함수 계산
y_linear = 2 * x_exp + 1
y_exp = np.exp(x_exp)
y_log = np.log(x_log)
y_sigmoid = 1 / (1 + np.exp(-x_sigmoid))
y_relu = np.maximum(0, x_relu)

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 선형 함수
axes[0, 0].plot(x_exp, y_linear, linewidth=2, color='blue')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Linear: f(x) = 2x + 1', fontsize=12)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

# 지수 함수
axes[0, 1].plot(x_exp, y_exp, linewidth=2, color='red')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Exponential: f(x) = eˣ', fontsize=12)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

# 로그 함수
axes[0, 2].plot(x_log, y_log, linewidth=2, color='green')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_title('Logarithm: f(x) = ln(x)', fontsize=12)
axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
axes[0, 2].axvline(x=0, color='k', linewidth=0.5)

# 시그모이드 함수
axes[1, 0].plot(x_sigmoid, y_sigmoid, linewidth=2, color='purple')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)', fontsize=12)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axhline(y=1, color='k', linewidth=0.5, linestyle='--')
axes[1, 0].axhline(y=0.5, color='r', linewidth=0.5, linestyle='--')
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

# ReLU 함수
axes[1, 1].plot(x_relu, y_relu, linewidth=2, color='orange')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('ReLU: f(x) = max(0, x)', fontsize=12)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

# 비교: Sigmoid vs ReLU
axes[1, 2].plot(x_sigmoid, y_sigmoid, linewidth=2, label='Sigmoid', color='purple')
axes[1, 2].plot(x_relu, y_relu / 5, linewidth=2, label='ReLU (scaled)', color='orange')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_title('Activation Functions', fontsize=12)
axes[1, 2].legend()
axes[1, 2].axhline(y=0, color='k', linewidth=0.5)
axes[1, 2].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('functions_overview.png', dpi=150, bbox_inches='tight')
print("함수 그래프 저장 완료!")
```

### 실습 3: 합성함수
```python
import numpy as np

# 기본 함수 정의
def f(x):
    """f(x) = x²"""
    return x**2

def g(x):
    """g(x) = x + 1"""
    return x + 1

def h(x):
    """h(x) = 2x"""
    return 2 * x

# 합성함수
def f_compose_g(x):
    """(f ∘ g)(x) = f(g(x))"""
    return f(g(x))

def g_compose_f(x):
    """(g ∘ f)(x) = g(f(x))"""
    return g(f(x))

def f_g_h(x):
    """f(g(h(x)))"""
    return f(g(h(x)))

# 테스트
x = 3
print("=== 합성함수 테스트 ===")
print(f"x = {x}")
print(f"f(x) = x² = {f(x)}")
print(f"g(x) = x + 1 = {g(x)}")
print(f"h(x) = 2x = {h(x)}")
print()
print(f"(f ∘ g)(x) = f(g(x)) = f({g(x)}) = {f_compose_g(x)}")
print(f"(g ∘ f)(x) = g(f(x)) = g({f(x)}) = {g_compose_f(x)}")
print(f"f(g(h(x))) = f(g({h(x)})) = f({g(h(x))}) = {f_g_h(x)}")
print()
print("⚠️ 합성함수는 순서가 중요! (f ∘ g) ≠ (g ∘ f)")
```

### 실습 4: 신경망 = 합성함수
```python
import numpy as np

# 간단한 2층 신경망
class TwoLayerNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 가중치 초기화
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def layer1(self, x):
        """첫 번째 층: f₁(x) = ReLU(W₁x + b₁)"""
        z = np.dot(x, self.W1) + self.b1
        return np.maximum(0, z)  # ReLU

    def layer2(self, h):
        """두 번째 층: f₂(h) = W₂h + b₂"""
        return np.dot(h, self.W2) + self.b2

    def forward(self, x):
        """전체 네트워크: y = f₂(f₁(x))"""
        h = self.layer1(x)  # 중간층
        y = self.layer2(h)  # 출력층
        return y

# 네트워크 생성
net = TwoLayerNetwork(input_dim=3, hidden_dim=4, output_dim=2)

# 테스트 입력
x = np.array([1.0, 2.0, 3.0])

print("=== 신경망 = 합성함수 ===")
print(f"입력 x: {x}")
print(f"입력 차원: {x.shape}")
print()

# 층별 출력 확인
h = net.layer1(x)
print(f"Layer 1 출력 (f₁(x)): {h}")
print(f"차원: {h.shape}")
print()

y = net.layer2(h)
print(f"Layer 2 출력 (f₂(f₁(x))): {y}")
print(f"차원: {y.shape}")
print()

# 전체 순전파
output = net.forward(x)
print(f"전체 네트워크 출력: {output}")
print("\n✅ 신경망 = 여러 함수를 합성한 것!")
```

---

## ✍️ 손 계산 연습

### 연습 1: 함수값 계산
다음 함수들의 값을 계산하세요:

1. f(x) = 3x + 2, f(4) = ?
   ```
   f(4) = 3(4) + 2 = 12 + 2 = 14
   ```

2. g(x) = x², g(-3) = ?
   ```
   g(-3) = (-3)² = 9
   ```

3. h(x) = 2ˣ, h(3) = ?
   ```
   h(3) = 2³ = 8
   ```

### 연습 2: 합성함수
f(x) = 2x, g(x) = x + 3일 때:

1. (f ∘ g)(5) = ?
   ```
   (f ∘ g)(5) = f(g(5))
              = f(5 + 3)
              = f(8)
              = 2(8)
              = 16
   ```

2. (g ∘ f)(5) = ?
   ```
   (g ∘ f)(5) = g(f(5))
              = g(2 × 5)
              = g(10)
              = 10 + 3
              = 13
   ```

3. (f ∘ g)(x) = ?
   ```
   (f ∘ g)(x) = f(g(x))
              = f(x + 3)
              = 2(x + 3)
              = 2x + 6
   ```

### 연습 3: 함수 그래프 스케치
다음 함수의 그래프 개형을 그려보세요:

1. f(x) = x (선형)
2. f(x) = eˣ (지수)
3. f(x) = ln(x) (로그)
4. f(x) = 1/(1+e⁻ˣ) (시그모이드)

---

## 🔗 LLM 연결점

### 1. Transformer = 거대한 합성함수

**Transformer 구조**:
```
Input → Embedding →
  → Layer 1 (Attention + FFN) →
  → Layer 2 (Attention + FFN) →
  → ... →
  → Layer N →
  → Output
```

**수학적 표현**:
```
y = f_N(...f_2(f_1(Embed(x))))
```

### 2. 활성화 함수의 역할

**선형 층만 쌓으면?**
```
y = W₂(W₁x) = (W₂W₁)x = W_totalx
```
→ 여러 층 = 하나의 선형 층 (의미 없음!)

**활성화 함수 추가**:
```
y = W₂(ReLU(W₁x))
```
→ 비선형성 도입! 복잡한 패턴 학습 가능

### 3. FFN (Feed-Forward Network)
**Transformer의 각 층**:
```python
def ffn(x):
    h = ReLU(W1 @ x + b1)  # f₁
    y = W2 @ h + b2         # f₂
    return y                # f₂(f₁(x))
```

---

## ✅ 체크포인트

- [ ] **함수를 "입력→계산→출력" 구조로 설명할 수 있나요?**

- [ ] **지수, 로그, 선형 함수의 그래프를 그릴 수 있나요?**

- [ ] **합성함수 (f ∘ g)(x)를 계산할 수 있나요?**

- [ ] **신경망이 왜 합성함수인지 이해했나요?**

- [ ] **활성화 함수의 역할을 설명할 수 있나요?**

---

## 🎓 핵심 요약

1. **함수**: 입력 → 규칙 → 출력
2. **주요 함수들**: 선형, 지수, 로그, 시그모이드, ReLU
3. **합성함수**: f(g(x)), 여러 함수를 순차 적용
4. **신경망**: 층층이 쌓인 합성함수
5. **활성화 함수**: 비선형성을 위해 필수

### 다음 학습
- **Day 5**: 집합과 논리
  - 데이터와 확률을 다루는 기초

---

**수고하셨습니다!** 🎉
