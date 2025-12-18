# Day 26: 경사하강법 (Gradient Descent) (1.5시간)

## 📚 학습 목표
- 경사하강법의 원리 완벽히 이해하기
- 학습률의 중요성 파악하기
- SGD, Momentum, Adam 등 변형 이해하기
- 실제 신경망 학습에 적용하기

---

## 🎯 강의 주제
**"기울기를 따라 최솟값 찾기"**

---

## 📖 핵심 개념

### 1. 경사하강법 (Gradient Descent)

**목표**: 함수 f(θ)의 최솟값 찾기

**알고리즘**:
```
1. θ를 임의로 초기화
2. 반복:
   a. 기울기 계산: g = ∇f(θ)
   b. 업데이트: θ = θ - α × g
   c. 수렴 확인
```

**직관**:
```
현재 위치에서 가장 가파른 내리막 방향(-∇f)으로 이동
```

---

### 2. 학습률 (Learning Rate)

**α (알파)**: 한 번에 얼마나 이동할지
```
θ_new = θ_old - α × ∇f(θ)
```

**너무 작으면**:
- 수렴이 매우 느림
- 계산 비용 증가

**너무 크면**:
- 진동 (oscillation)
- 발산 (divergence)

**적절한 값**:
- 빠르게 수렴
- 안정적

---

### 3. 경사하강법의 종류

**Batch Gradient Descent**:
```
전체 데이터셋으로 기울기 계산
g = (1/N) Σ ∇L(x_i, y_i)

장점: 정확한 기울기
단점: 느림, 메모리 많이 사용
```

**Stochastic Gradient Descent (SGD)**:
```
한 샘플로 기울기 계산
g = ∇L(x_i, y_i)

장점: 빠름, 메모리 적게 사용
단점: 노이즈 많음
```

**Mini-batch Gradient Descent**:
```
배치 크기만큼 평균
g = (1/B) Σ_{i in batch} ∇L(x_i, y_i)

실제로 가장 많이 사용! (B=32, 64, 128, ...)
```

---

### 4. 개선된 방법들

**Momentum**:
```
v = β × v + (1-β) × g
θ = θ - α × v

이전 방향을 기억 → 진동 감소
```

**Adam** (Adaptive Moment Estimation):
```
m = β₁ × m + (1-β₁) × g     # 1차 모멘트
v = β₂ × v + (1-β₂) × g²    # 2차 모멘트

θ = θ - α × m / (√v + ε)

가장 많이 사용됨!
```

---

## 💻 Python 실습

### 실습 1: 기본 경사하강법
```python
import numpy as np
import matplotlib.pyplot as plt

# 함수: f(x) = x² - 4x + 4 = (x-2)²
def f(x):
    return x**2 - 4*x + 4

def df(x):
    return 2*x - 4

# 경사하강법
def gradient_descent(start, learning_rate, iterations):
    x = start
    history = [x]

    for i in range(iterations):
        grad = df(x)
        x = x - learning_rate * grad
        history.append(x)

    return x, history

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

learning_rates = [0.1, 0.5, 1.1]
x_plot = np.linspace(-1, 5, 200)
y_plot = f(x_plot)

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]

    # 함수 그래프
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')

    # 경사하강법
    start = 4.0
    iterations = 20
    final_x, history = gradient_descent(start, lr, iterations)

    # 경로
    for i in range(len(history)-1):
        ax.arrow(history[i], f(history[i]),
                history[i+1] - history[i], f(history[i+1]) - f(history[i]),
                head_width=0.1, head_length=0.1, fc='red', ec='red',
                alpha=0.5)

    ax.scatter(history, [f(x) for x in history], c='red', s=50, zorder=5)
    ax.scatter([2], [0], c='green', s=200, marker='*', zorder=10, label='Minimum')

    ax.set_title(f'Learning Rate = {lr}', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 결과 출력
    print(f"\nLearning Rate = {lr}:")
    print(f"  Start: {start:.4f}")
    print(f"  Final: {final_x:.4f}")
    print(f"  Iterations: {len(history)-1}")

plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=150)
print("\n학습률 비교 시각화 저장 완료!")
```

### 실습 2: SGD vs Mini-batch
```python
import numpy as np

# 간단한 선형 회귀 데이터
np.random.seed(42)
N = 1000
X = np.random.randn(N, 1)
y = 3 * X + 2 + np.random.randn(N, 1) * 0.5  # y = 3x + 2 + noise

# 손실 함수: MSE
def loss(X, y, w, b):
    predictions = X * w + b
    return np.mean((predictions - y)**2)

# 기울기
def gradient(X, y, w, b):
    N = len(X)
    predictions = X * w + b
    dw = (2/N) * np.sum(X * (predictions - y))
    db = (2/N) * np.sum(predictions - y)
    return dw, db

print("=== SGD vs Mini-batch ===\n")

# Batch GD
w_batch, b_batch = 0.0, 0.0
lr = 0.01
epochs = 50

for epoch in range(epochs):
    dw, db = gradient(X, y, w_batch, b_batch)
    w_batch -= lr * dw
    b_batch -= lr * db

print("Batch Gradient Descent:")
print(f"  w = {w_batch:.4f}, b = {b_batch:.4f}")
print(f"  True: w = 3.0, b = 2.0\n")

# Mini-batch GD
w_mini, b_mini = 0.0, 0.0
batch_size = 32
lr = 0.01
epochs = 50

for epoch in range(epochs):
    # 데이터 섞기
    indices = np.random.permutation(N)

    for i in range(0, N, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        dw, db = gradient(X_batch, y_batch, w_mini, b_mini)
        w_mini -= lr * dw
        b_mini -= lr * db

print("Mini-batch Gradient Descent (batch_size=32):")
print(f"  w = {w_mini:.4f}, b = {b_mini:.4f}")
```

### 실습 3: Momentum vs Adam
```python
import numpy as np
import matplotlib.pyplot as plt

# 함수: Rosenbrock (최적화가 어려운 함수)
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# SGD
def sgd(start, lr, iterations):
    theta = np.array(start, dtype=float)
    path = [theta.copy()]

    for _ in range(iterations):
        grad = grad_rosenbrock(*theta)
        theta = theta - lr * grad
        path.append(theta.copy())

    return path

# Momentum
def momentum(start, lr, beta, iterations):
    theta = np.array(start, dtype=float)
    v = np.zeros_like(theta)
    path = [theta.copy()]

    for _ in range(iterations):
        grad = grad_rosenbrock(*theta)
        v = beta * v + grad
        theta = theta - lr * v
        path.append(theta.copy())

    return path

# Adam (간소화)
def adam(start, lr, beta1, beta2, iterations):
    theta = np.array(start, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    epsilon = 1e-8

    for t in range(1, iterations+1):
        grad = grad_rosenbrock(*theta)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(theta.copy())

    return path

# 실행
start = [-1.0, -0.5]
iterations = 200

path_sgd = sgd(start, lr=0.001, iterations=iterations)
path_momentum = momentum(start, lr=0.001, beta=0.9, iterations=iterations)
path_adam = adam(start, lr=0.01, beta1=0.9, beta2=0.999, iterations=iterations)

# 시각화
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(12, 10))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)

path_sgd = np.array(path_sgd)
path_momentum = np.array(path_momentum)
path_adam = np.array(path_adam)

plt.plot(path_sgd[:, 0], path_sgd[:, 1], 'r-', label='SGD', linewidth=2)
plt.plot(path_momentum[:, 0], path_momentum[:, 1], 'g-', label='Momentum', linewidth=2)
plt.plot(path_adam[:, 0], path_adam[:, 1], 'b-', label='Adam', linewidth=2)

plt.scatter([1], [1], c='yellow', s=300, marker='*', zorder=10, label='Optimum (1, 1)')
plt.scatter([start[0]], [start[1]], c='red', s=200, zorder=10, label='Start')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimizer Comparison: Rosenbrock Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150)
print("최적화 알고리즘 비교 시각화 저장 완료!")

print("\n최종 위치:")
print(f"  SGD:      {path_sgd[-1]}")
print(f"  Momentum: {path_momentum[-1]}")
print(f"  Adam:     {path_adam[-1]}")
print(f"  Optimum:  [1.0, 1.0]")
```

---

## ✍️ 손 계산 연습

### 연습 1: 경사하강법 2단계
```
f(x) = x² - 6x + 9
f'(x) = 2x - 6

시작: x = 0, α = 0.5

Step 1:
  g = 2(0) - 6 = -6
  x = 0 - 0.5(-6) = 3

Step 2:
  g = 2(3) - 6 = 0
  x = 3 - 0.5(0) = 3

수렴! (최솟값: x=3)
```

### 연습 2: Momentum 1단계
```
θ = [1, 2], g = [4, 6]
v = [0, 0], β = 0.9, α = 0.1

v_new = 0.9[0, 0] + [4, 6] = [4, 6]
θ_new = [1, 2] - 0.1[4, 6] = [0.6, 1.4]
```

---

## 🔗 LLM 연결점

### 1. 실제 학습
```python
# PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()  # 기울기 계산
    optimizer.step()  # 경사하강법!
    optimizer.zero_grad()
```

### 2. 학습률 스케줄링
```
초기: 큰 학습률 (빠른 수렴)
후기: 작은 학습률 (미세 조정)

Warmup: 처음엔 천천히
Decay: 점점 줄이기
```

### 3. Gradient Accumulation
```
메모리 부족 시:
여러 배치의 기울기를 누적 후 업데이트

실질적 배치 크기 증가 효과
```

---

## ✅ 체크포인트

- [ ] **경사하강법의 원리를 설명할 수 있나요?**

- [ ] **학습률이 왜 중요한지 이해했나요?**

- [ ] **SGD, Momentum, Adam의 차이를 아나요?**

- [ ] **신경망 학습에서의 역할을 이해했나요?**

---

## 🎓 핵심 요약

1. **원리**: θ = θ - α∇f(θ)
2. **학습률**: 너무 크거나 작으면 안 됨
3. **Mini-batch**: 실제 많이 사용
4. **Adam**: 가장 안정적이고 효과적

### 다음 학습
- **Day 27**: 미적분 종합 복습

---

**수고하셨습니다!** 🎉

**경사하강법은 모든 딥러닝의 핵심입니다!**
