# Day 24: 연쇄법칙 (Chain Rule) (1.5시간)

## 📚 학습 목표
- 연쇄법칙의 정의와 사용법 완벽히 이해하기
- 합성함수의 미분 익히기
- **Backpropagation의 수학적 기초 완성하기** ⭐
- 다층 신경망의 기울기 계산 이해하기

---

## 🎯 강의 주제
**"신경망의 영혼 - Backpropagation의 수학"**

---

## 📖 핵심 개념

### 1. 연쇄법칙 (Chain Rule)

**합성함수**:
```
y = f(g(x))

예: y = (x² + 1)³
    f(u) = u³
    g(x) = x² + 1
```

**연쇄법칙**:
```
dy/dx = dy/du × du/dx

= f'(g(x)) × g'(x)
```

**예시**:
```
y = (x² + 1)³

u = x² + 1  →  du/dx = 2x
y = u³      →  dy/du = 3u²

dy/dx = 3u² × 2x
      = 3(x² + 1)² × 2x
      = 6x(x² + 1)²
```

---

### 2. 다변수 연쇄법칙

**경로가 여러 개**:
```
z = f(x, y)
x = g(t)
y = h(t)

dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
```

**신경망 예시**:
```
L = loss(y)
y = f(z)
z = wx + b

dL/dw = (dL/dy) × (dy/dz) × (dz/dw)
       ↑         ↑         ↑
    손실 기울기  활성화    입력
```

---

### 3. Backpropagation의 수학

**단순한 신경망**:
```
입력 → 은닉층 → 출력 → 손실

x → z₁ = W₁x + b₁ → a₁ = σ(z₁)
  → z₂ = W₂a₁ + b₂ → y = σ(z₂)
  → L = (y - target)²
```

**역전파 (연쇄법칙 적용)**:
```
dL/dW₂ = dL/dy × dy/dz₂ × dz₂/dW₂

dL/dW₁ = dL/dy × dy/dz₂ × dz₂/da₁ × da₁/dz₁ × dz₁/dW₁
```

**핵심 통찰**:
```
뒤에서 앞으로 (back) 기울기를 전파(propagation)!
```

---

### 4. 계산 그래프

**그래프 표현**:
```
x ─→ [×w] ─→ [+b] ─→ [σ] ─→ y ─→ [L]
      ↓        ↓       ↓       ↓      ↓
     dw       db      da      dy     dL
```

**Forward pass**: 왼쪽 → 오른쪽 (값 계산)
**Backward pass**: 오른쪽 → 왼쪽 (기울기 계산)

---

## 💻 Python 실습

### 실습 1: 연쇄법칙 기초
```python
import numpy as np

# 합성함수: y = (x² + 1)³
def g(x):
    return x**2 + 1

def f(u):
    return u**3

def composite(x):
    return f(g(x))

# 도함수
def g_prime(x):
    return 2 * x

def f_prime(u):
    return 3 * u**2

def composite_prime(x):
    """연쇄법칙"""
    u = g(x)
    return f_prime(u) * g_prime(x)

# 테스트
x = 2.0

print("=== 연쇄법칙 ===\n")
print(f"x = {x}\n")

# Forward
u = g(x)
y = f(u)

print("Forward:")
print(f"  u = g(x) = x² + 1 = {u}")
print(f"  y = f(u) = u³ = {y}\n")

# Derivative
du_dx = g_prime(x)
dy_du = f_prime(u)
dy_dx = composite_prime(x)

print("연쇄법칙:")
print(f"  du/dx = 2x = {du_dx}")
print(f"  dy/du = 3u² = {dy_du}")
print(f"  dy/dx = (dy/du)(du/dx) = {dy_dx}\n")

# 수치 미분으로 확인
h = 1e-5
numerical = (composite(x + h) - composite(x)) / h
print(f"수치 미분: {numerical:.6f}")
print(f"연쇄법칙: {dy_dx:.6f}")
print(f"일치: {np.isclose(numerical, dy_dx)}")
```

### 실습 2: 2층 신경망 Backpropagation
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 네트워크 구조
input_size = 2
hidden_size = 3
output_size = 1

# 초기화
np.random.seed(42)
W1 = np.random.randn(hidden_size, input_size) * 0.1
b1 = np.zeros(hidden_size)
W2 = np.random.randn(output_size, hidden_size) * 0.1
b2 = np.zeros(output_size)

# 입력과 목표
x = np.array([1.0, 2.0])
target = 0.8

print("=== 2층 신경망 Backpropagation ===\n")
print(f"입력: {x}")
print(f"목표: {target}\n")

# ===== Forward Pass =====
print("Forward Pass:")

z1 = W1 @ x + b1
a1 = sigmoid(z1)
print(f"  z1 = W1·x + b1")
print(f"  a1 = σ(z1) = {a1}")

z2 = W2 @ a1 + b2
y = sigmoid(z2)
print(f"  z2 = W2·a1 + b2")
print(f"  y = σ(z2) = {y}")

loss = (y - target)**2
print(f"  loss = (y - target)² = {loss}\n")

# ===== Backward Pass (연쇄법칙!) =====
print("Backward Pass (Chain Rule):\n")

# 출력층
dL_dy = 2 * (y - target)
dy_dz2 = sigmoid_derivative(z2)
dL_dz2 = dL_dy * dy_dz2

print(f"1. 출력층:")
print(f"   dL/dy = 2(y - target) = {dL_dy}")
print(f"   dy/dz2 = σ'(z2) = {dy_dz2}")
print(f"   dL/dz2 = dL/dy × dy/dz2 = {dL_dz2}\n")

# W2, b2의 기울기
dz2_dW2 = a1
dL_dW2 = np.outer(dL_dz2, dz2_dW2)
dL_db2 = dL_dz2

print(f"2. 출력층 파라미터:")
print(f"   dL/dW2 =\n{dL_dW2}")
print(f"   dL/db2 = {dL_db2}\n")

# 은닉층으로 전파
dL_da1 = W2.T @ dL_dz2
da1_dz1 = sigmoid_derivative(z1)
dL_dz1 = dL_da1 * da1_dz1

print(f"3. 은닉층으로 전파:")
print(f"   dL/da1 = W2^T × dL/dz2 = {dL_da1}")
print(f"   da1/dz1 = σ'(z1) = {da1_dz1}")
print(f"   dL/dz1 = dL/da1 ⊙ da1/dz1 = {dL_dz1}\n")

# W1, b1의 기울기
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

print(f"4. 은닉층 파라미터:")
print(f"   dL/dW1 =\n{dL_dW1}")
print(f"   dL/db1 = {dL_db1}\n")

# Gradient descent
lr = 0.1
W1 -= lr * dL_dW1
b1 -= lr * dL_db1
W2 -= lr * dL_dW2
b2 -= lr * dL_db2

# 새로운 loss
z1_new = W1 @ x + b1
a1_new = sigmoid(z1_new)
z2_new = W2 @ a1_new + b2
y_new = sigmoid(z2_new)
loss_new = (y_new - target)**2

print(f"업데이트 후:")
print(f"  loss: {loss[0]:.6f} → {loss_new[0]:.6f}")
print(f"  개선: {loss[0] - loss_new[0]:.6f} ✓")
```

### 실습 3: 계산 그래프 시각화
```python
import numpy as np

class ComputationNode:
    """계산 그래프 노드"""
    def __init__(self, name):
        self.name = name
        self.value = None
        self.grad = 0

    def __repr__(self):
        return f"{self.name}(val={self.value:.4f}, grad={self.grad:.4f})"

# 간단한 예: y = x * w + b
x = ComputationNode("x")
w = ComputationNode("w")
b = ComputationNode("b")
mul = ComputationNode("x*w")
y = ComputationNode("y")
L = ComputationNode("L")

# Forward
x.value = 2.0
w.value = 3.0
b.value = 1.0
target = 10.0

mul.value = x.value * w.value
y.value = mul.value + b.value
L.value = (y.value - target)**2

print("=== 계산 그래프 ===\n")
print("Forward Pass:")
print(f"  {x}")
print(f"  {w}")
print(f"  {b}")
print(f"  {mul}")
print(f"  {y}")
print(f"  {L}\n")

# Backward
L.grad = 1.0  # dL/dL
y.grad = L.grad * 2 * (y.value - target)  # dL/dy
b.grad = y.grad * 1.0  # dL/db
mul.grad = y.grad * 1.0  # dL/d(mul)
w.grad = mul.grad * x.value  # dL/dw
x.grad = mul.grad * w.value  # dL/dx

print("Backward Pass (Chain Rule):")
print(f"  {L}")
print(f"  {y}")
print(f"  {b}")
print(f"  {mul}")
print(f"  {w}")
print(f"  {x}\n")

print("기울기:")
print(f"  dL/dw = {w.grad:.4f}")
print(f"  dL/db = {b.grad:.4f}")
```

---

## ✍️ 손 계산 연습

### 연습 1: 기본 연쇄법칙
```
y = (2x + 1)²

u = 2x + 1  →  du/dx = 2
y = u²      →  dy/du = 2u

dy/dx = 2u × 2 = 4u = 4(2x + 1)

x = 1:  dy/dx = 4(3) = 12
```

### 연습 2: 3단계 합성
```
y = e^(x²)

u = x²     →  du/dx = 2x
y = e^u    →  dy/du = e^u

dy/dx = e^u × 2x = 2x e^(x²)
```

### 연습 3: 간단한 Backprop
```
L = (y - 1)²
y = σ(wx)

x = 2, w = 0.5, y = σ(1) ≈ 0.731

dL/dy = 2(y - 1) = 2(-0.269) = -0.538
dy/dw = σ'(wx) × x = 0.196 × 2 = 0.392
dL/dw = -0.538 × 0.392 = -0.211
```

---

## 🔗 LLM 연결점

### 1. Transformer의 Backpropagation
```
Attention → FFN → LayerNorm → ...

각 층마다 연쇄법칙 적용
수십~수백 층을 거슬러 올라감
```

### 2. 기울기 소실/폭주
```
연쇄법칙: 여러 미분값을 곱함

<1 값들을 계속 곱하면 → 0 (소실)
>1 값들을 계속 곱하면 → ∞ (폭주)

→ Residual Connection, LayerNorm으로 해결
```

### 3. Automatic Differentiation
```
PyTorch, TensorFlow:
자동으로 연쇄법칙 적용

tensor.backward() → 모든 기울기 계산!
```

---

## ✅ 체크포인트

- [ ] **연쇄법칙을 설명할 수 있나요?**

- [ ] **합성함수의 미분을 계산할 수 있나요?**

- [ ] **Backpropagation이 연쇄법칙임을 이해했나요?**

- [ ] **계산 그래프에서 기울기를 역전파할 수 있나요?**

---

## 🎓 핵심 요약

1. **연쇄법칙**: dy/dx = (dy/du)(du/dx)
2. **합성함수**: f(g(x))' = f'(g(x)) × g'(x)
3. **Backprop**: 출력 → 입력으로 기울기 전파
4. **핵심**: 모든 신경망 학습의 기초!

### 다음 학습
- **Day 25**: 편미분과 기울기

---

**수고하셨습니다!** 🎉

**연쇄법칙은 Backpropagation의 본질입니다!**
