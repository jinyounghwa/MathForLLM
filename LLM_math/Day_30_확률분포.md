# Day 30: 확률분포 (1.5시간)

## 📚 학습 목표
- 확률분포의 개념 이해하기
- 정규분포의 성질 파악하기
- 평균과 분산 계산하기

---

## 🎯 강의 주제
**"확률이 어떻게 분포하는가?"**

---

## 📖 핵심 개념

### 1. 확률분포
```
P(X = x): X가 x일 확률

이산: P(X = 1), P(X = 2), ...
연속: 확률밀도함수 f(x)
```

### 2. 정규분포 (Gaussian)
```
N(μ, σ²)

μ: 평균 (mean)
σ²: 분산 (variance)
σ: 표준편차 (standard deviation)

f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
```

### 3. 평균과 분산
```
평균: E[X] = Σ x × P(X=x)
분산: Var(X) = E[(X-μ)²]
표준편차: σ = √Var(X)
```

---

## 💻 Python 실습

```python
import numpy as np
import matplotlib.pyplot as plt

# 정규분포 생성
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 10000)

print("=== 정규분포 ===")
print(f"이론: μ={mu}, σ={sigma}")
print(f"샘플: μ={np.mean(samples):.4f}, σ={np.std(samples):.4f}\n")

# 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='샘플')

x = np.linspace(-4, 4, 100)
y = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))
plt.plot(x, y, 'r-', linewidth=2, label='이론')

plt.xlabel('x')
plt.ylabel('확률밀도')
plt.title('정규분포 N(0, 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('normal_distribution.png', dpi=150)
print("시각화 저장!")
```

---

## 🔗 LLM 연결점

### 임베딩 초기화
```python
# PyTorch
embedding = nn.Embedding(vocab_size, embed_dim)
# 내부적으로 N(0, 1)로 초기화

가중치도 정규분포로 초기화!
```

---

## 🎓 핵심 요약

1. **정규분포**: N(μ, σ²)
2. **평균**: 중심
3. **분산**: 퍼진 정도

### 다음 학습
- **Day 31**: 중간 복습

---

**정규분포는 자연과 AI에서 가장 중요합니다!**
