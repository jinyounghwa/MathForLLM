# Day 21: 극한과 연속성 (1시간)

## 📚 학습 목표
- 극한의 개념 이해하기
- 연속함수의 정의 파악하기
- 미분의 기초 다지기

---

## 🎯 강의 주제
**"한없이 가까워진다는 것"**

---

## 📖 핵심 개념

### 1. 극한 (Limit)

**정의**:
```
lim_{x→a} f(x) = L

x가 a에 한없이 가까워질 때,
f(x)가 L에 한없이 가까워진다
```

**기호 읽기**:
```
lim (리밋, 극한)
x→a (x가 a로 간다)
```

**예시**:
```
f(x) = 2x + 1

lim_{x→3} f(x) = lim_{x→3} (2x + 1)
                = 2(3) + 1
                = 7
```

---

### 2. 극한의 성질

**합의 극한**:
```
lim_{x→a} [f(x) + g(x)] = lim_{x→a} f(x) + lim_{x→a} g(x)
```

**곱의 극한**:
```
lim_{x→a} [f(x) × g(x)] = lim_{x→a} f(x) × lim_{x→a} g(x)
```

**스칼라배**:
```
lim_{x→a} [k × f(x)] = k × lim_{x→a} f(x)
```

---

### 3. 불연속 vs 연속

**불연속의 예**:
```
f(x) = 1/x

x→0일 때 극한이 존재하지 않음
(양쪽에서 다가갈 때 +∞, -∞)
```

**연속의 예**:
```
f(x) = x²

모든 점에서 연속
```

---

### 4. 연속성 (Continuity)

**정의**: 함수 f(x)가 x = a에서 연속
```
1. f(a)가 정의됨
2. lim_{x→a} f(x)가 존재
3. lim_{x→a} f(x) = f(a)
```

**직관적 의미**:
```
그래프를 펜을 떼지 않고 그릴 수 있다
```

---

### 5. 미분으로 가는 길

**평균 변화율**:
```
(f(b) - f(a)) / (b - a)

구간 [a, b]에서 f의 평균 변화율
```

**순간 변화율** (미분):
```
lim_{h→0} (f(a+h) - f(a)) / h

x = a에서의 순간 변화율
```

---

## 💻 Python 실습

### 실습 1: 극한 시각화
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """함수 f(x) = x²"""
    return x**2

# 극한 확인: lim_{x→2} x² = 4
a = 2
x_values = [2.1, 2.01, 2.001, 2.0001, 2.00001]

print("=== 극한 확인: lim_{x→2} x² ===\n")
print("x가 2에 가까워질 때 f(x)의 값:")
for x in x_values:
    fx = f(x)
    print(f"  x = {x:>10.5f}  →  f(x) = {fx:.10f}")

print(f"\n→ 극한값: {f(2)}")

# 시각화
x = np.linspace(0, 4, 200)
y = f(x)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
plt.scatter([2], [4], color='red', s=200, zorder=5, label='lim point (2, 4)')
plt.axhline(4, color='r', linestyle='--', alpha=0.5)
plt.axvline(2, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Limit: lim_{x→2} x² = 4', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('limit_visualization.png', dpi=150)
print("\n시각화 저장 완료!")
```

### 실습 2: 연속 vs 불연속
```python
import numpy as np
import matplotlib.pyplot as plt

# 연속 함수
def continuous(x):
    return x**2

# 불연속 함수
def discontinuous(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1

x_cont = np.linspace(-2, 2, 200)
y_cont = continuous(x_cont)

x_disc = np.linspace(-2, 2, 200)
y_disc = np.array([discontinuous(xi) for xi in x_disc])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 연속 함수
ax1.plot(x_cont, y_cont, 'b-', linewidth=2)
ax1.scatter([0], [0], color='red', s=100, zorder=5)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('Continuous: f(x) = x²', fontsize=14)

# 불연속 함수
ax2.plot(x_disc, y_disc, 'r-', linewidth=2)
ax2.scatter([0], [0], color='blue', s=100, zorder=5)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('Discontinuous: f(x) = sign(x)', fontsize=14)

plt.tight_layout()
plt.savefig('continuity.png', dpi=150)
print("연속성 시각화 저장 완료!")
```

### 실습 3: 평균 vs 순간 변화율
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

# a = 2에서의 변화율
a = 2

# 평균 변화율 (h를 점점 작게)
h_values = [1, 0.5, 0.1, 0.01, 0.001]

print("=== 평균 변화율 → 순간 변화율 ===\n")
print(f"함수: f(x) = x², 점: x = {a}\n")

for h in h_values:
    avg_rate = (f(a + h) - f(a)) / h
    print(f"h = {h:>6.3f}  →  평균 변화율 = {avg_rate:.6f}")

print(f"\n→ 순간 변화율 (미분): {2*a}")

# 시각화
x = np.linspace(0, 4, 200)
y = f(x)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')

# 점 a
plt.scatter([a], [f(a)], color='red', s=200, zorder=5, label=f'Point (a={a})')

# 할선 (평균 변화율)
for h in [1, 0.5, 0.1]:
    x_secant = [a, a + h]
    y_secant = [f(a), f(a + h)]
    plt.plot(x_secant, y_secant, '--', alpha=0.5, label=f'h={h}')

# 접선 (순간 변화율)
slope = 2 * a  # f'(2) = 4
x_tangent = np.linspace(1, 3, 100)
y_tangent = f(a) + slope * (x_tangent - a)
plt.plot(x_tangent, y_tangent, 'r-', linewidth=2, label='Tangent (derivative)')

plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Average Rate → Instantaneous Rate', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('rate_of_change.png', dpi=150)
print("\n변화율 시각화 저장 완료!")
```

---

## ✍️ 손 계산 연습

### 연습 1: 극한 계산
```
lim_{x→3} (x² + 2x)

= 3² + 2(3)
= 9 + 6
= 15
```

### 연습 2: 평균 변화율
```
f(x) = x²
a = 1, b = 3

평균 변화율 = (f(3) - f(1)) / (3 - 1)
            = (9 - 1) / 2
            = 4
```

### 연습 3: 순간 변화율 추정
```
f(x) = x², a = 2

h = 0.1:  (f(2.1) - f(2)) / 0.1 = (4.41 - 4) / 0.1 = 4.1
h = 0.01: (f(2.01) - f(2)) / 0.01 = 4.01
h = 0.001: ≈ 4.001

→ 순간 변화율 ≈ 4
```

---

## 🔗 LLM 연결점

### 1. 손실 함수의 연속성
```
손실 함수 L(θ)는 연속이어야
경사하강법이 제대로 작동함
```

### 2. 활성화 함수
```
ReLU: 불연속 미분 (x=0에서)
Sigmoid, Tanh: 모든 점에서 연속

연속성 → 안정적인 학습
```

### 3. 미분 가능성
```
Backpropagation:
연쇄법칙으로 기울기 계산

연속 + 미분 가능 → 필수!
```

---

## ✅ 체크포인트

- [ ] **극한의 의미를 설명할 수 있나요?**

- [ ] **연속함수의 조건을 이해했나요?**

- [ ] **평균 변화율과 순간 변화율의 차이를 아나요?**

- [ ] **미분이 왜 필요한지 감이 잡히나요?**

---

## 🎓 핵심 요약

1. **극한**: lim_{x→a} f(x) = L
2. **연속**: lim_{x→a} f(x) = f(a)
3. **평균 변화율**: Δf / Δx
4. **순간 변화율**: lim_{h→0} Δf / Δx = 미분

### 다음 학습
- **Day 22-23**: 미분 (도함수)
  - 미분의 정의와 계산 규칙

---

**수고하셨습니다!** 🎉

**극한은 미적분의 출발점입니다!**
