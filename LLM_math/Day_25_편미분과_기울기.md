# Day 25: 편미분과 기울기 (1.5시간)

## 📚 학습 목표
- 편미분의 개념 이해하기
- 기울기 벡터 (Gradient) 계산하기
- 다변수 함수의 최적화 이해하기
- 신경망 파라미터 업데이트 원리 파악하기

---

## 🎯 강의 주제
**"여러 변수일 때의 미분"**

---

## 📖 핵심 개념

### 1. 편미분 (Partial Derivative)

**정의**: 한 변수만 변화시키고 나머지는 고정
```
∂f/∂x: x에 대한 편미분 (y는 상수 취급)
∂f/∂y: y에 대한 편미분 (x는 상수 취급)
```

**예시**:
```
f(x, y) = x²y + 3x + 2y

∂f/∂x = 2xy + 3  (y를 상수로)
∂f/∂y = x² + 2   (x를 상수로)
```

**계산**:
```
(x, y) = (2, 3):

∂f/∂x = 2(2)(3) + 3 = 15
∂f/∂y = 2² + 2 = 6
```

---

### 2. 기울기 벡터 (Gradient)

**정의**: 모든 편미분을 모은 벡터
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**2변수 예시**:
```
f(x, y) = x² + y²

∇f = [∂f/∂x, ∂f/∂y]
   = [2x, 2y]
```

**의미**:
```
∇f는 f가 가장 빠르게 증가하는 방향을 가리킴
```

---

### 3. 기울기의 기하학

**등고선과 기울기**:
```
f(x, y) = x² + y² = c  (원)

∇f는 등고선에 수직!
```

**방향 도함수**:
```
v⃗ 방향으로의 변화율:
D_v⃗ f = ∇f · v̂  (v̂: 단위 벡터)
```

**최대 변화율**:
```
∇f 방향으로 가장 빠르게 증가
-∇f 방향으로 가장 빠르게 감소
```

---

### 4. 신경망에서의 기울기

**손실 함수**:
```
L(w₁, w₂, ..., wₙ, b₁, b₂, ..., bₘ)

모든 파라미터에 대한 편미분 필요!
```

**기울기 벡터**:
```
∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ,
      ∂L/∂b₁, ∂L/∂b₂, ..., ∂L/∂bₘ]
```

**업데이트**:
```
θ_new = θ_old - α × ∇L(θ)

α: 학습률
-∇L: 손실이 감소하는 방향
```

---

## 💻 Python 실습

### 실습 1: 편미분 기초
```python
import numpy as np

# 함수: f(x, y) = x²y + 3x + 2y
def f(x, y):
    return x**2 * y + 3*x + 2*y

# 편미분 (해석적)
def df_dx(x, y):
    return 2*x*y + 3

def df_dy(x, y):
    return x**2 + 2

# 수치 편미분
def numerical_partial_x(f, x, y, h=1e-5):
    return (f(x+h, y) - f(x, y)) / h

def numerical_partial_y(f, x, y, h=1e-5):
    return (f(x, y+h) - f(x, y)) / h

# 테스트
x, y = 2.0, 3.0

print("=== 편미분 ===\n")
print(f"f(x, y) = x²y + 3x + 2y")
print(f"점: ({x}, {y})\n")

# 해석적 편미분
partial_x = df_dx(x, y)
partial_y = df_dy(x, y)

print("해석적 편미분:")
print(f"  ∂f/∂x = {partial_x}")
print(f"  ∂f/∂y = {partial_y}\n")

# 수치 편미분
numerical_x = numerical_partial_x(f, x, y)
numerical_y = numerical_partial_y(f, x, y)

print("수치 편미분:")
print(f"  ∂f/∂x ≈ {numerical_x:.6f}")
print(f"  ∂f/∂y ≈ {numerical_y:.6f}\n")

# 기울기 벡터
gradient = np.array([partial_x, partial_y])
print(f"기울기 벡터: ∇f = {gradient}")
```

### 실습 2: 기울기와 등고선
```python
import numpy as np
import matplotlib.pyplot as plt

# 함수: f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

# 기울기
def gradient_f(x, y):
    return np.array([2*x, 2*y])

# 그리드 생성
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 시각화
fig, ax = plt.subplots(figsize=(10, 10))

# 등고선
contours = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
ax.clabel(contours, inline=True, fontsize=10)

# 기울기 벡터 (일부 점에서)
points = [(1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (0, 2)]

for px, py in points:
    grad = gradient_f(px, py)
    # 기울기 방향
    ax.arrow(px, py, grad[0]*0.3, grad[1]*0.3,
             head_width=0.2, head_length=0.15, fc='red', ec='red',
             linewidth=2)
    # 점
    ax.scatter([px], [py], color='red', s=100, zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('f(x,y) = x² + y²: Contours and Gradients', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('gradient_contours.png', dpi=150)
print("등고선과 기울기 시각화 저장 완료!")
```

### 실습 3: 경사하강법 시뮬레이션
```python
import numpy as np
import matplotlib.pyplot as plt

# 함수: f(x, y) = (x-2)² + (y-3)²
# 최솟값: (2, 3)
def f(x, y):
    return (x - 2)**2 + (y - 3)**2

def gradient_f(x, y):
    return np.array([2*(x - 2), 2*(y - 3)])

# 경사하강법
learning_rate = 0.1
iterations = 50

# 시작점
theta = np.array([0.0, 0.0])

# 경로 기록
path = [theta.copy()]
losses = [f(*theta)]

print("=== 경사하강법 ===\n")
print(f"학습률: {learning_rate}")
print(f"반복: {iterations}\n")
print(f"시작점: {theta}\n")

for i in range(iterations):
    grad = gradient_f(*theta)
    theta = theta - learning_rate * grad

    path.append(theta.copy())
    losses.append(f(*theta))

    if i % 10 == 0 or i == iterations - 1:
        print(f"Step {i:2d}: θ = [{theta[0]:.4f}, {theta[1]:.4f}], "
              f"loss = {losses[-1]:.6f}")

print(f"\n최적점: [2.0, 3.0]")
print(f"도달점: [{theta[0]:.4f}, {theta[1]:.4f}]")

# 시각화
path = np.array(path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 경로 시각화
x = np.linspace(-1, 4, 100)
y = np.linspace(-1, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2,
         markersize=5, label='Gradient Descent Path')
ax1.scatter([2], [3], color='green', s=300, marker='*',
            zorder=5, label='Optimum')
ax1.scatter([0], [0], color='red', s=200,
            zorder=5, label='Start')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Gradient Descent Path', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 손실 감소
ax2.plot(losses, 'b-', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss Decrease', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150)
print("\n경사하강법 시각화 저장 완료!")
```

---

## ✍️ 손 계산 연습

### 연습 1: 편미분
```
f(x, y) = 3x² + 2xy + y²

∂f/∂x = 6x + 2y
∂f/∂y = 2x + 2y

(1, 2):
∂f/∂x = 6(1) + 2(2) = 10
∂f/∂y = 2(1) + 2(2) = 6

∇f = [10, 6]
```

### 연습 2: 기울기
```
f(x, y) = x² + 4y²

∇f = [2x, 8y]

(2, 1):  ∇f = [4, 8]
```

### 연습 3: 경사하강법 1단계
```
f(x, y) = x² + y²
∇f = [2x, 2y]

시작: (4, 3)
학습률: α = 0.1

∇f(4, 3) = [8, 6]

θ_new = [4, 3] - 0.1[8, 6]
      = [4, 3] - [0.8, 0.6]
      = [3.2, 2.4]
```

---

## 🔗 LLM 연결점

### 1. 파라미터 업데이트
```python
# PyTorch
optimizer.step()

내부적으로:
for param in model.parameters():
    param = param - lr * param.grad
    # param.grad = ∂L/∂param (편미분!)
```

### 2. Adam Optimizer
```
m_t = β₁ m_{t-1} + (1-β₁) ∇L     # 1차 모멘트
v_t = β₂ v_{t-1} + (1-β₂) (∇L)²  # 2차 모멘트

θ_t = θ_{t-1} - α × m_t / √v_t

기울기 벡터 ∇L를 적응적으로 조정!
```

### 3. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(params, max_norm)

||∇L|| > max_norm이면:
∇L = ∇L × (max_norm / ||∇L||)
```

---

## ✅ 체크포인트

- [ ] **편미분을 계산할 수 있나요?**

- [ ] **기울기 벡터의 의미를 이해했나요?**

- [ ] **경사하강법이 왜 -∇L 방향인지 아나요?**

- [ ] **신경망 파라미터 업데이트 원리를 이해했나요?**

---

## 🎓 핵심 요약

1. **편미분**: ∂f/∂x (다른 변수 고정)
2. **기울기**: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]
3. **방향**: ∇f = 최대 증가 방향
4. **최적화**: θ - α∇L = 손실 감소

### 다음 학습
- **Day 26**: 경사하강법 심화

---

**수고하셨습니다!** 🎉

**기울기는 최적화의 나침반입니다!**
