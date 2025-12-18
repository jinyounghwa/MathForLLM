# Day 5: 집합과 논리 (1시간)

## 📚 학습 목표
- 집합의 기본 개념과 연산 이해하기
- 논리 연산과 집합 연산의 관계 파악하기
- 조건부 확률의 기초 다지기

---

## 🎯 강의 주제
**"데이터와 연산을 다루기 위한 언어"**

---

## 📖 핵심 개념

### 1. 집합 (Set)

#### 1.1 집합의 정의
**서로 다른 원소들의 모임**

```
A = {1, 2, 3, 4, 5}
B = {2, 4, 6, 8}
```

**표기법**:
- 원소 나열: A = {1, 2, 3}
- 조건 제시: A = {x | x는 양의 정수, x ≤ 5}
- 읽기: "x는 A의 원소다" → x ∈ A

**특징**:
- 순서 없음: {1, 2, 3} = {3, 1, 2}
- 중복 없음: {1, 1, 2} = {1, 2}
- 공집합: ∅ 또는 {}

---

#### 1.2 집합의 기본 연산

**1. 합집합 (Union): A ∪ B**
"A 또는 B에 속하는 원소"

```
A = {1, 2, 3}
B = {3, 4, 5}
A ∪ B = {1, 2, 3, 4, 5}
```

**2. 교집합 (Intersection): A ∩ B**
"A와 B 모두에 속하는 원소"

```
A = {1, 2, 3}
B = {3, 4, 5}
A ∩ B = {3}
```

**3. 차집합 (Difference): A - B**
"A에는 속하지만 B에는 속하지 않는 원소"

```
A = {1, 2, 3}
B = {3, 4, 5}
A - B = {1, 2}
```

**4. 여집합 (Complement): Aᶜ**
"전체 집합 U에서 A가 아닌 원소"

```
U = {1, 2, 3, 4, 5}
A = {1, 2, 3}
Aᶜ = {4, 5}
```

**5. 부분집합 (Subset): A ⊆ B**
"A의 모든 원소가 B에 속함"

```
A = {1, 2}
B = {1, 2, 3, 4}
A ⊆ B (참)
```

---

### 2. 논리 (Logic)

#### 2.1 명제 (Proposition)
**참(True) 또는 거짓(False)을 판단할 수 있는 문장**

```
참인 명제: "2 + 2 = 4"
거짓인 명제: "3 > 5"
명제 아님: "너 몇 살이야?" (질문)
```

#### 2.2 논리 연산자

**1. AND (논리곱): ∧**
"둘 다 참일 때만 참"

```
진리표:
P | Q | P ∧ Q
T | T |   T
T | F |   F
F | T |   F
F | F |   F
```

**2. OR (논리합): ∨**
"하나라도 참이면 참"

```
진리표:
P | Q | P ∨ Q
T | T |   T
T | F |   T
F | T |   T
F | F |   F
```

**3. NOT (부정): ¬**
"참과 거짓을 뒤집기"

```
P | ¬P
T |  F
F |  T
```

**4. Implication (함의): P → Q**
"P이면 Q이다"

```
진리표:
P | Q | P → Q
T | T |   T
T | F |   F
F | T |   T
F | F |   T
```

---

#### 2.3 논리와 집합의 관계

**드모르간의 법칙 (De Morgan's Laws)**:
```
¬(P ∧ Q) = (¬P) ∨ (¬Q)
¬(P ∨ Q) = (¬P) ∧ (¬Q)

집합으로:
(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
```

**대응 관계**:
```
논리 AND (∧)  ↔ 집합 교집합 (∩)
논리 OR (∨)   ↔ 집합 합집합 (∪)
논리 NOT (¬)  ↔ 집합 여집합 (ᶜ)
```

---

### 3. 조건부 확률의 기초

#### 3.1 조건부 확률
**어떤 사건 B가 일어났을 때, A가 일어날 확률**

```
P(A|B) = P(A ∩ B) / P(B)
```

**직관**:
- "B가 일어났다"는 것을 알고 있음
- 이제 전체 샘플 공간이 B로 축소됨
- 그 안에서 A가 일어날 확률

**예시**:
```
주사위를 던졌을 때:
A = {짝수가 나옴} = {2, 4, 6}
B = {4 이상이 나옴} = {4, 5, 6}

P(A|B) = "4 이상이 나왔을 때, 짝수일 확률"
       = P(A ∩ B) / P(B)
       = P({4, 6}) / P({4, 5, 6})
       = (2/6) / (3/6)
       = 2/3
```

#### 3.2 독립 사건
**B가 일어나든 말든, A의 확률이 변하지 않음**

```
P(A|B) = P(A)  ⟺  A와 B가 독립
```

**예시**:
- 동전 두 번 던지기: 첫 번째 결과가 두 번째에 영향 없음
- LLM 토큰 예측: 이전 토큰들이 다음 토큰에 영향 있음 (종속!)

---

## 💻 Python 실습

### 실습 1: 집합 연산
```python
# Python의 set 자료구조
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

print("=== 집합 연산 ===")
print(f"A = {A}")
print(f"B = {B}")
print()

# 합집합
print(f"A ∪ B = {A | B}")
print(f"또는: {A.union(B)}")
print()

# 교집합
print(f"A ∩ B = {A & B}")
print(f"또는: {A.intersection(B)}")
print()

# 차집합
print(f"A - B = {A - B}")
print(f"또는: {A.difference(B)}")
print()

# 대칭 차집합 (XOR)
print(f"A △ B = {A ^ B}")
print(f"또는: {A.symmetric_difference(B)}")
print()

# 부분집합 확인
C = {2, 3}
print(f"C = {C}")
print(f"C ⊆ A? {C.issubset(A)}")
print(f"A ⊇ C? {A.issuperset(C)}")
```

### 실습 2: 논리 연산
```python
import numpy as np

# 진리표 생성
def truth_table():
    print("=== AND (∧) 진리표 ===")
    print("P\tQ\tP ∧ Q")
    for p in [True, False]:
        for q in [True, False]:
            result = p and q
            print(f"{p}\t{q}\t{result}")

    print("\n=== OR (∨) 진리표 ===")
    print("P\tQ\tP ∨ Q")
    for p in [True, False]:
        for q in [True, False]:
            result = p or q
            print(f"{p}\t{q}\t{result}")

    print("\n=== NOT (¬) 진리표 ===")
    print("P\t¬P")
    for p in [True, False]:
        result = not p
        print(f"{p}\t{result}")

truth_table()

# 드모르간 법칙 검증
print("\n=== 드모르간 법칙 검증 ===")
for p in [True, False]:
    for q in [True, False]:
        lhs = not (p and q)
        rhs = (not p) or (not q)
        print(f"P={p}, Q={q}: ¬(P∧Q) = {lhs}, (¬P)∨(¬Q) = {rhs}, 같음? {lhs == rhs}")
```

### 실습 3: 조건부 확률
```python
import numpy as np
from collections import Counter

# 주사위 시뮬레이션
np.random.seed(42)
n_trials = 100000

# 주사위 던지기
dice_rolls = np.random.randint(1, 7, n_trials)

# 사건 정의
even = dice_rolls % 2 == 0  # 짝수
greater_than_3 = dice_rolls > 3  # 4 이상

# 확률 계산
P_even = np.mean(even)
P_gt3 = np.mean(greater_than_3)
P_even_and_gt3 = np.mean(even & greater_than_3)

# 조건부 확률
P_even_given_gt3 = P_even_and_gt3 / P_gt3

print("=== 조건부 확률 시뮬레이션 ===")
print(f"시행 횟수: {n_trials:,}")
print()
print(f"P(짝수) = {P_even:.4f} (이론값: 0.5000)")
print(f"P(4 이상) = {P_gt3:.4f} (이론값: 0.5000)")
print(f"P(짝수 ∩ 4 이상) = {P_even_and_gt3:.4f} (이론값: 0.3333)")
print()
print(f"P(짝수 | 4 이상) = {P_even_given_gt3:.4f} (이론값: 0.6667)")
print()
print("해석: 4 이상이 나왔다는 조건 하에, 짝수일 확률은 2/3")
print("      {4, 5, 6} 중 짝수는 {4, 6} → 2/3")
```

### 실습 4: LLM 토큰 필터링 (집합 활용)
```python
# LLM 어휘 집합
vocab = {"안녕", "하세요", "감사", "합니다", ".", "!", "?"}
stopwords = {".", "!", "?"}

print("=== LLM 토큰 필터링 ===")
print(f"전체 어휘: {vocab}")
print(f"불용어: {stopwords}")
print()

# 불용어 제거
content_words = vocab - stopwords
print(f"내용어 (불용어 제거): {content_words}")
print()

# 실제 문장의 토큰
sentence_tokens = {"안녕", "하세요", "!"}
print(f"문장 토큰: {sentence_tokens}")

# 어휘에 있는 토큰만 (교집합)
valid_tokens = sentence_tokens & vocab
print(f"유효한 토큰: {valid_tokens}")

# 미등록 토큰 (차집합)
oov_tokens = sentence_tokens - vocab
print(f"OOV 토큰: {oov_tokens}")
```

---

## ✍️ 손 계산 연습

### 연습 1: 집합 연산
다음을 계산하세요:

```
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7}
C = {5, 6, 7, 8}
```

1. A ∪ B = ?
   ```
   답: {1, 2, 3, 4, 5, 6, 7}
   ```

2. A ∩ B = ?
   ```
   답: {4, 5}
   ```

3. A - B = ?
   ```
   답: {1, 2, 3}
   ```

4. (A ∩ B) ∪ C = ?
   ```
   답: {4, 5} ∪ {5, 6, 7, 8} = {4, 5, 6, 7, 8}
   ```

### 연습 2: 논리 연산
다음의 진릿값을 구하세요:

```
P = True, Q = False
```

1. P ∧ Q = ?
   ```
   답: True ∧ False = False
   ```

2. P ∨ Q = ?
   ```
   답: True ∨ False = True
   ```

3. ¬P = ?
   ```
   답: ¬True = False
   ```

4. ¬(P ∧ Q) = ?
   ```
   답: ¬False = True
   ```

5. (¬P) ∨ (¬Q) = ?
   ```
   답: False ∨ True = True
   ```

### 연습 3: 조건부 확률
카드 52장 중:
- 하트: 13장
- 그림 카드 (J, Q, K): 12장
- 하트 그림 카드: 3장

1. P(하트) = ?
   ```
   답: 13/52 = 1/4
   ```

2. P(그림 카드) = ?
   ```
   답: 12/52 = 3/13
   ```

3. P(하트 | 그림 카드) = ?
   ```
   답: P(하트 ∩ 그림) / P(그림)
     = (3/52) / (12/52)
     = 3/12
     = 1/4
   ```

---

## 🔗 LLM 연결점

### 1. 어휘 집합 (Vocabulary Set)
```python
vocab = {"안녕", "하세요", "감사", "합니다", ...}  # 32,000개
```

**토큰화 과정**:
```
문장: "안녕하세요!"
토큰: ["안녕", "하세요", "!"]

각 토큰 ∈ vocab?
- "안녕" ∈ vocab ✓
- "하세요" ∈ vocab ✓
- "!" ∈ vocab ✓
```

### 2. Attention Mask (집합 연산)
**어텐션 마스크 = 어떤 토큰에 주목할지 결정**

```python
# 문장 토큰: ["안녕", "하세요", "<PAD>", "<PAD>"]
# 실제 토큰: {0, 1}
# 패딩 토큰: {2, 3}

attention_mask = [1, 1, 0, 0]  # 1 = 주목, 0 = 무시
```

**집합으로 표현**:
```
A = {토큰을 주목할 위치} = {0, 1}
Aᶜ = {무시할 위치} = {2, 3}
```

### 3. 조건부 확률과 LLM
**LLM의 본질 = 조건부 확률 모델**

```
P(w_next | w_1, w_2, ..., w_n)
```

"이전 단어들이 주어졌을 때, 다음 단어의 확률"

**예시**:
```
문맥: "오늘 날씨가 정말"
P("좋다" | "오늘 날씨가 정말") = 0.7
P("나쁘다" | "오늘 날씨가 정말") = 0.2
P("맛있다" | "오늘 날씨가 정말") = 0.001
```

---

## ✅ 체크포인트

- [ ] **집합 기호 (∪, ∩, ⊆)를 이해했나요?**

- [ ] **논리 연산 (∧, ∨, ¬)의 진리표를 만들 수 있나요?**

- [ ] **드모르간 법칙을 설명할 수 있나요?**

- [ ] **논리 연산과 집합 연산의 대응을 이해했나요?**

- [ ] **조건부 확률 P(A|B)의 의미를 설명할 수 있나요?**

- [ ] **LLM이 조건부 확률 모델임을 이해했나요?**

---

## 🎓 핵심 요약

1. **집합**: 원소들의 모임, 연산 (∪, ∩, -, ᶜ)
2. **논리**: 명제와 연산 (∧, ∨, ¬)
3. **대응 관계**: 논리 ↔ 집합
4. **조건부 확률**: P(A|B) = P(A∩B) / P(B)
5. **LLM 연결**: 어휘 집합, Attention Mask, 조건부 확률 모델

### 다음 학습
- **Day 6-7**: 변수, 방정식, 연립방정식
  - 모르는 수를 찾기
  - 행렬 표현으로 연결

---

**수고하셨습니다!** 🎉
