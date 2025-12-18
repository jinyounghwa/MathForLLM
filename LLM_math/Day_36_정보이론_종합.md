# Day 36: 정보이론 종합 (2시간)

## 📚 학습 목표
- 엔트로피, 상호정보량, 정보 이득의 연결 이해하기
- 정보이론의 통합적 그림 보기

---

## 🎯 강의 주제
**"정보이론의 전체 지도"**

---

## 📖 개념 연결

### 전체 구조
```
확률 P(X)
    ↓
엔트로피 H(X) (불확실성)
    ↓
조건부 엔트로피 H(X|Y)
    ↓
상호정보량 I(X;Y) = H(X) - H(X|Y)
    ↓
정보 이득 IG = H(부모) - H(자식들)
    ↓
의사결정, 압축, 학습
```

---

## 💻 종합 프로젝트

```python
import numpy as np

class InformationTheory:
    """정보이론 도구 모음"""

    @staticmethod
    def entropy(probs):
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def cross_entropy(p, q):
        return -np.sum(p * np.log2(q + 1e-10))

    @staticmethod
    def kl_divergence(p, q):
        """KL Divergence: D(P||Q)"""
        return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

    @staticmethod
    def mutual_information(joint):
        """상호정보량"""
        p_x = joint.sum(axis=1)
        p_y = joint.sum(axis=0)

        mi = 0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i,j] > 0:
                    mi += joint[i,j] * np.log2(
                        joint[i,j] / (p_x[i] * p_y[j])
                    )
        return mi

# 사용 예
it = InformationTheory()

p = np.array([0.5, 0.3, 0.2])
print(f"H(P) = {it.entropy(p):.4f} bits")

q = np.array([0.4, 0.4, 0.2])
print(f"CE(P, Q) = {it.cross_entropy(p, q):.4f}")
print(f"KL(P||Q) = {it.kl_divergence(p, q):.4f}")
```

---

## 🔗 LLM 총정리

### 정보이론 → LLM
| 개념 | LLM 적용 |
|------|----------|
| 엔트로피 | Perplexity |
| Cross Entropy | 손실 함수 |
| KL Divergence | 분포 비교 |
| 정보 이득 | BPE 토크나이저 |
| 상호정보량 | Attention 해석 |

---

## 🎓 핵심 요약

**정보이론은 LLM의 수학적 언어**

1. 불확실성 측정: 엔트로피
2. 모델 평가: Cross Entropy
3. 토큰화: 정보 이득
4. 학습: 손실 최소화

### 다음 학습
- **Day 37-38**: 최종 복습

---

**정보이론 마스터 완료!**
