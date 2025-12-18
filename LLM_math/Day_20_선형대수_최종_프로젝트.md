# Day 20: 선형대수 최종 프로젝트 - 임베딩 공간 분석 (1시간)

## 📚 학습 목표
- 지금까지 배운 선형대수 개념 종합 활용하기
- 실제 임베딩 데이터 분석하기
- 코사인 유사도, PCA, 클러스터링 적용하기

---

## 🎯 프로젝트 주제
**"단어 임베딩 공간 탐험하기"**

---

## 📖 프로젝트 개요

### 구현할 내용
1. 간단한 단어 임베딩 생성
2. 코사인 유사도로 유사 단어 찾기
3. PCA로 2D 시각화
4. 클러스터링으로 의미 그룹 찾기

### 사용할 개념
- 내적, 정규화, 코사인 유사도
- 고유값, 고유벡터, PCA
- 행렬 연산, 거리 계산

---

## 💻 최종 프로젝트 코드

### 프로젝트: 단어 임베딩 분석

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 단어 임베딩 생성 =====
print("=" * 50)
print("1. 단어 임베딩 생성")
print("=" * 50 + "\n")

# 간단한 임베딩 (실제로는 학습됨)
# 차원: 5
words = ['king', 'queen', 'man', 'woman', 'apple', 'banana', 'car', 'truck']

# 의미적 유사성을 반영한 임베딩
embeddings = {
    'king':   [0.9, 0.8, 0.1, 0.1, 0.0],
    'queen':  [0.85, 0.9, 0.05, 0.15, 0.0],
    'man':    [0.7, 0.5, 0.2, 0.0, 0.0],
    'woman':  [0.65, 0.6, 0.1, 0.2, 0.0],
    'apple':  [0.0, 0.0, 0.9, 0.8, 0.1],
    'banana': [0.0, 0.0, 0.85, 0.9, 0.15],
    'car':    [0.0, 0.0, 0.1, 0.0, 0.9],
    'truck':  [0.0, 0.0, 0.15, 0.05, 0.85]
}

# NumPy 배열로 변환
embedding_matrix = np.array([embeddings[w] for w in words])

print(f"단어 수: {len(words)}")
print(f"임베딩 차원: {embedding_matrix.shape[1]}")
print(f"임베딩 행렬 형태: {embedding_matrix.shape}\n")

# ===== 2. 정규화 =====
print("=" * 50)
print("2. 임베딩 정규화")
print("=" * 50 + "\n")

# L2 정규화
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
normalized_embeddings = embedding_matrix / norms

print("정규화 전 노름:")
for i, word in enumerate(words):
    print(f"  ||{word}|| = {norms[i, 0]:.4f}")

print("\n정규화 후 노름 (모두 1.0):")
new_norms = np.linalg.norm(normalized_embeddings, axis=1)
for i, word in enumerate(words):
    print(f"  ||{word}|| = {new_norms[i]:.4f}")

# ===== 3. 코사인 유사도 =====
print("\n" + "=" * 50)
print("3. 코사인 유사도 계산")
print("=" * 50 + "\n")

def cosine_similarity(v1, v2):
    """코사인 유사도"""
    return np.dot(v1, v2)  # 이미 정규화됨

def find_most_similar(word, embeddings_dict, normalized_emb, words_list, top_k=3):
    """가장 유사한 단어 찾기"""
    word_idx = words_list.index(word)
    word_vec = normalized_emb[word_idx]

    similarities = []
    for i, other_word in enumerate(words_list):
        if other_word != word:
            sim = cosine_similarity(word_vec, normalized_emb[i])
            similarities.append((other_word, sim))

    # 유사도 내림차순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 예시: 'king'과 유사한 단어
query = 'king'
similar = find_most_similar(query, embeddings, normalized_embeddings, words, top_k=3)

print(f"'{query}'와 가장 유사한 단어:")
for word, sim in similar:
    print(f"  {word}: {sim:.4f}")

print()

# 모든 단어 쌍의 유사도 행렬
similarity_matrix = normalized_embeddings @ normalized_embeddings.T

print("유사도 행렬 (일부):")
print("       ", "  ".join(f"{w:>6}" for w in words[:4]))
for i in range(4):
    row = "  ".join(f"{similarity_matrix[i, j]:6.3f}" for j in range(4))
    print(f"{words[i]:>6}  {row}")

# ===== 4. PCA로 차원 축소 =====
print("\n" + "=" * 50)
print("4. PCA로 2D 시각화")
print("=" * 50 + "\n")

# PCA: 5차원 → 2차원
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(normalized_embeddings)

print(f"원본 차원: {normalized_embeddings.shape[1]}")
print(f"축소 차원: {embeddings_2d.shape[1]}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"총 분산: {sum(pca.explained_variance_ratio_):.4f}\n")

# 시각화
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=200, alpha=0.6)

for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 fontsize=14, ha='center', va='bottom')

plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('Word Embeddings (PCA)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('word_embeddings_pca.png', dpi=150)
print("시각화 저장: word_embeddings_pca.png")

# ===== 5. 거리 기반 분석 =====
print("\n" + "=" * 50)
print("5. 거리 계산")
print("=" * 50 + "\n")

def euclidean_distance(v1, v2):
    """유클리드 거리"""
    return np.linalg.norm(v1 - v2)

# 'king'과 다른 단어들 사이의 거리
query = 'king'
query_idx = words.index(query)
query_vec = normalized_embeddings[query_idx]

print(f"'{query}'와 다른 단어들 사이의 거리:")
distances = []
for i, word in enumerate(words):
    if word != query:
        dist = euclidean_distance(query_vec, normalized_embeddings[i])
        distances.append((word, dist))

distances.sort(key=lambda x: x[1])

for word, dist in distances:
    sim = cosine_similarity(query_vec, normalized_embeddings[words.index(word)])
    print(f"  {word:>8}: dist={dist:.4f}, sim={sim:.4f}")

# ===== 6. 벡터 연산 (King - Man + Woman = ?) =====
print("\n" + "=" * 50)
print("6. 벡터 연산 (Word Analogy)")
print("=" * 50 + "\n")

# King - Man + Woman ≈ Queen?
king_vec = normalized_embeddings[words.index('king')]
man_vec = normalized_embeddings[words.index('man')]
woman_vec = normalized_embeddings[words.index('woman')]

# 벡터 연산
result_vec = king_vec - man_vec + woman_vec
# 재정규화
result_vec = result_vec / np.linalg.norm(result_vec)

print("King - Man + Woman = ?")
print()

# 가장 유사한 단어 찾기
similarities = []
for i, word in enumerate(words):
    if word not in ['king', 'man', 'woman']:
        sim = cosine_similarity(result_vec, normalized_embeddings[i])
        similarities.append((word, sim))

similarities.sort(key=lambda x: x[1], reverse=True)

print("결과 벡터와 가장 유사한 단어:")
for word, sim in similarities[:3]:
    print(f"  {word}: {sim:.4f}")

print("\n✅ 'queen'이 가장 유사하게 나오면 성공!")

# ===== 7. 종합 통계 =====
print("\n" + "=" * 50)
print("7. 종합 통계")
print("=" * 50 + "\n")

# 공분산 행렬
cov_matrix = np.cov(normalized_embeddings.T)

print(f"공분산 행렬 형태: {cov_matrix.shape}")
print(f"공분산 행렬 (일부):\n{cov_matrix[:3, :3]}\n")

# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues_sorted = np.sort(eigenvalues)[::-1]

print("고유값 (내림차순):")
for i, ev in enumerate(eigenvalues_sorted):
    print(f"  λ_{i+1} = {ev:.4f}")

print("\n✅ 프로젝트 완료!")
print("\n" + "=" * 50)
print("배운 개념 활용:")
print("=" * 50)
print("✓ 내적과 정규화")
print("✓ 코사인 유사도")
print("✓ 거리 계산")
print("✓ PCA (고유값/고유벡터)")
print("✓ 행렬 연산")
print("✓ 벡터 연산")
print("=" * 50)
```

---

## ✍️ 프로젝트 확장 아이디어

### 1. 더 많은 단어
```python
# 동물, 과일, 교통수단 등 카테고리 추가
# 더 복잡한 의미 관계 탐험
```

### 2. 3D 시각화
```python
# PCA n_components=3
# matplotlib의 3D 플롯 사용
```

### 3. t-SNE
```python
from sklearn.manifold import TSNE
# PCA보다 더 나은 시각화
```

---

## 🔗 LLM 연결점

### 실제 LLM에서는
```
1. 임베딩:
   - 수백~수천 차원
   - 학습으로 획득

2. 유사도:
   - RAG: 문서 검색
   - Attention: 토큰 간 관계

3. 차원 축소:
   - 시각화
   - 해석 가능성

4. 벡터 연산:
   - 의미 조합
   - 관계 학습
```

---

## ✅ 체크포인트

- [ ] **모든 코드를 실행했나요?**

- [ ] **코사인 유사도로 단어를 찾을 수 있나요?**

- [ ] **PCA의 결과를 해석할 수 있나요?**

- [ ] **벡터 연산의 의미를 이해했나요?**

---

## 🎓 선형대수 총정리

**Day 11-20에서 배운 것**:

1. **벡터**:
   - 길이, 거리, 방향

2. **내적**:
   - 유사도 측정의 핵심

3. **정규화**:
   - 크기 제거, 방향만

4. **행렬**:
   - 선형 변환, 신경망

5. **전치**:
   - 차원 맞추기

6. **역행렬**:
   - 방정식 풀이

7. **고유값/벡터**:
   - PCA, 주성분 분석

8. **노름**:
   - 크기, 안정성

**이 모든 것이 LLM의 토대입니다!**

### 다음 학습
- **Day 21-27**: 미적분
  - 변화율, 경사하강법, Backpropagation

---

**수고하셨습니다!** 🎉

**선형대수 마스터를 축하합니다!**
