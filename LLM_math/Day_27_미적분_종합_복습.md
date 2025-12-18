# Day 27: 미적분 종합 복습 (1시간)

## 📚 학습 목표
- Day 21-26의 핵심 개념 총정리
- 미적분과 신경망의 연결 확인
- Backpropagation 전체 흐름 이해
- 다음 단계(확률) 준비

---

## 🎯 강의 주제
**"미적분으로 신경망을 이해하다"**

---

## 📖 핵심 개념 정리

### 1. 극한과 연속 (Day 21)
```
lim_{x→a} f(x) = L

연속: lim_{x→a} f(x) = f(a)

→ 미분 가능의 전제 조건
```

---

### 2. 미분 (Day 22-23)
```
f'(x) = lim_{h→0} (f(x+h) - f(x)) / h

기본 공식:
- (x^n)' = nx^{n-1}
- (e^x)' = e^x
- (ln x)' = 1/x

활성화 함수:
- σ'(x) = σ(x)(1-σ(x))
- tanh'(x) = 1 - tanh²(x)
- ReLU'(x) = {1 if x>0, 0 otherwise}
```

---

### 3. 연쇄법칙 (Day 24)
```
dy/dx = (dy/du)(du/dx)

Backpropagation의 핵심!

dL/dw = dL/dy × dy/dz × dz/dw
```

---

### 4. 편미분과 기울기 (Day 25)
```
∂f/∂x: x에 대한 편미분

∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]

기울기 = 가장 빠른 증가 방향
```

---

### 5. 경사하강법 (Day 26)
```
θ = θ - α∇L(θ)

Adam:
m = β₁m + (1-β₁)g
v = β₂v + (1-β₂)g²
θ = θ - α × m/√v
```

---

## 🔗 전체 연결: 신경망 학습

### Forward Pass
```python
# 입력
x = [x₁, x₂, ..., xₙ]

# Layer 1
z₁ = W₁·x + b₁
a₁ = σ(z₁)

# Layer 2
z₂ = W₂·a₁ + b₂
y = σ(z₂)

# 손실
L = (y - target)²
```

### Backward Pass (연쇄법칙)
```python
# 출력층
dL/dy = 2(y - target)
dy/dz₂ = σ'(z₂)
dL/dz₂ = dL/dy × dy/dz₂

dL/dW₂ = dL/dz₂ × a₁ᵀ  (외적)
dL/db₂ = dL/dz₂

# 은닉층
dL/da₁ = W₂ᵀ × dL/dz₂
da₁/dz₁ = σ'(z₁)
dL/dz₁ = dL/da₁ ⊙ da₁/dz₁  (원소별 곱)

dL/dW₁ = dL/dz₁ × xᵀ
dL/db₁ = dL/dz₁
```

### 업데이트 (경사하강법)
```python
W₁ = W₁ - α × dL/dW₁
b₁ = b₁ - α × dL/db₁
W₂ = W₂ - α × dL/dW₂
b₂ = b₂ - α × dL/db₂
```

---

## 💻 종합 실습

### 전체 흐름 구현
```python
import numpy as np

class TwoLayerNetwork:
    """2층 신경망 (완전 구현)"""

    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (He 초기화)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        """Forward pass"""
        # Layer 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.y = self.sigmoid(self.z2)

        return self.y

    def backward(self, x, target):
        """Backward pass (Backpropagation)"""
        # 출력층
        dL_dy = 2 * (self.y - target)
        dy_dz2 = self.sigmoid_derivative(self.z2)
        dL_dz2 = dL_dy * dy_dz2

        # 기울기 계산
        self.dW2 = np.outer(dL_dz2, self.a1)
        self.db2 = dL_dz2

        # 은닉층으로 전파
        dL_da1 = self.W2.T @ dL_dz2
        da1_dz1 = self.sigmoid_derivative(self.z1)
        dL_dz1 = dL_da1 * da1_dz1

        # 기울기 계산
        self.dW1 = np.outer(dL_dz1, x)
        self.db1 = dL_dz1

    def update(self, learning_rate):
        """파라미터 업데이트 (경사하강법)"""
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def train_step(self, x, target, learning_rate):
        """한 스텝 학습"""
        # Forward
        y = self.forward(x)
        loss = np.sum((y - target)**2)

        # Backward
        self.backward(x, target)

        # Update
        self.update(learning_rate)

        return loss

# 사용 예시
print("=== 2층 신경망 종합 실습 ===\n")

# 네트워크 생성
net = TwoLayerNetwork(input_size=3, hidden_size=4, output_size=1)

# 학습 데이터
X_train = [
    np.array([0.1, 0.2, 0.3]),
    np.array([0.4, 0.5, 0.6]),
    np.array([0.7, 0.8, 0.9]),
]

y_train = [
    np.array([0.2]),
    np.array([0.6]),
    np.array([0.9]),
]

# 학습
epochs = 100
learning_rate = 0.1

print("학습 시작...\n")

for epoch in range(epochs):
    total_loss = 0

    for x, target in zip(X_train, y_train):
        loss = net.train_step(x, target, learning_rate)
        total_loss += loss

    if epoch % 20 == 0 or epoch == epochs - 1:
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")

print("\n학습 완료!\n")

# 테스트
print("테스트:")
for i, (x, target) in enumerate(zip(X_train, y_train)):
    y_pred = net.forward(x)
    print(f"  입력: {x}")
    print(f"  목표: {target[0]:.4f}, 예측: {y_pred[0]:.4f}")
    print()
```

---

## ✍️ 자가 진단 문제

### 문제 1: 미분
```
f(x) = 3x² + 2e^x

f'(x) = ?
```

<details>
<summary>정답</summary>

```
f'(x) = 6x + 2e^x
```
</details>

### 문제 2: 연쇄법칙
```
y = (2x + 1)³

dy/dx = ?
```

<details>
<summary>정답</summary>

```
u = 2x + 1  →  du/dx = 2
y = u³      →  dy/du = 3u²

dy/dx = 3u² × 2 = 6(2x + 1)²
```
</details>

### 문제 3: 편미분
```
f(x, y) = x²y + 3x

∂f/∂x = ?
∂f/∂y = ?
```

<details>
<summary>정답</summary>

```
∂f/∂x = 2xy + 3
∂f/∂y = x²
```
</details>

### 문제 4: 경사하강법
```
f(x) = x² - 4x
f'(x) = 2x - 4

시작: x = 0, α = 0.5
1단계 후 x = ?
```

<details>
<summary>정답</summary>

```
g = 2(0) - 4 = -4
x = 0 - 0.5(-4) = 2
```
</details>

---

## 🎓 미적분 → 신경망 매핑

| 미적분 개념 | 신경망 적용 |
|------------|-------------|
| 함수 | 모델 (입력→출력) |
| 미분 | 기울기 계산 |
| 연쇄법칙 | Backpropagation |
| 편미분 | 파라미터별 기울기 |
| 기울기 | ∇L (손실의 기울기) |
| 경사하강법 | 최적화 (학습) |

---

## ✅ 최종 체크포인트

- [ ] **미분의 정의를 설명할 수 있나요?**

- [ ] **연쇄법칙으로 합성함수를 미분할 수 있나요?**

- [ ] **Backpropagation의 원리를 이해했나요?**

- [ ] **기울기 벡터의 의미를 아나요?**

- [ ] **경사하강법으로 최적화할 수 있나요?**

- [ ] **신경망 학습의 전체 흐름을 설명할 수 있나요?**

---

## 🎓 핵심 요약

**미적분이 신경망의 언어입니다!**

1. **미분**: 변화율
2. **연쇄법칙**: Backpropagation
3. **기울기**: 최적 방향
4. **경사하강법**: 학습

**이제 확률로 넘어갑니다!**

### 다음 학습
- **Day 28-38**: 확률과 정보이론
  - 불확실성, 엔트로피, 정보 이득

---

**수고하셨습니다!** 🎉

**미적분 마스터를 축하합니다!**
