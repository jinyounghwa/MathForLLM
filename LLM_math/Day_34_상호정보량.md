# Day 34: 상호정보량 (Mutual Information) (1.5시간)

## 📚 학습 목표
- 상호정보량의 정의 이해하기
- 두 변수의 연관성 측정하기

---

## 🎯 강의 주제
**"두 변수가 얼마나 연관있나?"**

---

## 📖 핵심 개념

### 상호정보량
```
I(X; Y) = H(X) - H(X|Y)

"Y를 알면 X의 불확실성이 얼마나 줄어드는가?"
```

**다른 표현**:
```
I(X; Y) = Σ P(x,y) log [P(x,y) / (P(x)P(y))]

독립이면: I(X; Y) = 0
```

---

## 💻 Python 실습

```python
import numpy as np

# 예: 날씨(X)와 우산(Y)
# X: {맑음, 비}, Y: {있음, 없음}

# 동시 확률
joint = np.array([[0.3, 0.1],   # P(맑음, 있음/없음)
                  [0.05, 0.55]]) # P(비, 있음/없음)

p_x = joint.sum(axis=1)  # 날씨
p_y = joint.sum(axis=0)  # 우산

# 엔트로피
h_x = -np.sum(p_x * np.log2(p_x))
h_y = -np.sum(p_y * np.log2(p_y))

# 상호정보량
mi = 0
for i in range(2):
    for j in range(2):
        if joint[i,j] > 0:
            mi += joint[i,j] * np.log2(joint[i,j] / (p_x[i] * p_y[j]))

print(f"I(날씨; 우산) = {mi:.4f} bits")
print("→ 날씨를 알면 우산 여부 불확실성이 줄어듦!")
```

---

## 🎓 핵심 요약

**상호정보량**: 변수 간 종속성 측정

### 다음 학습
- **Day 35**: 정보 이득

---

**변수들이 얼마나 서로 알려주는가!**
