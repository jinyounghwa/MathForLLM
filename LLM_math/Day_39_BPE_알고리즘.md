# Day 39: BPE 알고리즘 (Byte Pair Encoding) (2시간)

## 📚 학습 목표
- **BPE 알고리즘의 원리 완벽히 이해하기** ⭐
- 정보 이득 최대화와의 연결 파악하기
- 간단한 BPE 구현하기

---

## 🎯 강의 주제
**"텍스트를 효율적으로 나누기"**

---

## 📖 핵심 개념

### 1. BPE란?

**문제**:
```
단어 기반: 어휘가 너무 큼
문자 기반: 시퀀스가 너무 김

→ 서브워드(subword) 필요!
```

**BPE 아이디어**:
```
자주 나오는 바이트(문자) 쌍을 하나의 토큰으로 병합
```

---

### 2. BPE 알고리즘

**단계**:
```
1. 텍스트를 문자로 분리
2. 가장 빈번한 바이트 쌍 찾기
3. 그 쌍을 새 토큰으로 병합
4. 어휘 크기에 도달할 때까지 반복
```

**예시**:
```
원본: "low low low lowest"

초기: l o w   l o w   l o w   l o w e s t

Step 1: 'l o'가 가장 빈번 → 'lo'
        lo w   lo w   lo w   lo w e s t

Step 2: 'lo w'가 가장 빈번 → 'low'
        low   low   low   low e s t

Step 3: 'low'가 가장 빈번 (더 이상 병합 안 함)

최종 어휘: {l, o, w, e, s, t, lo, low, lowest}
```

---

### 3. 정보 이득과의 연결

**정보 이득 관점**:
```
자주 나오는 쌍 병합 = 압축률 향상
= 엔트로피 감소
= 정보 이득 최대화!
```

---

## 💻 Python 실습

### 실습 1: BPE 구현
```python
import re
from collections import Counter

class SimpleBPE:
    """간단한 BPE 토크나이저"""

    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.bpe_codes = []

    def get_stats(self, vocab):
        """바이트 쌍 빈도 계산"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """어휘에서 쌍 병합"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, corpus):
        """BPE 학습"""
        # 단어별 빈도
        words = corpus.lower().split()
        vocab = Counter(words)

        # 문자로 분리
        vocab = {' '.join(word): freq for word, freq in vocab.items()}

        print("=== BPE 학습 ===\n")
        print(f"초기 어휘: {len(vocab)}개 단어\n")

        for i in range(self.num_merges):
            pairs = self.get_stats(vocab)

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.bpe_codes.append(best)

            print(f"Step {i+1}: '{best[0]}' + '{best[1]}' → '{''.join(best)}' "
                  f"(빈도: {pairs[best]})")

        print(f"\n학습 완료! {len(self.bpe_codes)}개 병합\n")

        return vocab

# 사용 예
corpus = "low low low low lowest lower"
bpe = SimpleBPE(num_merges=5)
final_vocab = bpe.train(corpus)

print("최종 어휘:")
for word, freq in sorted(final_vocab.items(), key=lambda x: -x[1]):
    print(f"  '{word}': {freq}")
```

### 실습 2: 압축률 계산
```python
import numpy as np

def calculate_compression_ratio(original, encoded):
    """압축률 계산"""
    original_size = len(original.replace(' ', ''))
    encoded_size = len(encoded.split())
    ratio = original_size / encoded_size
    return ratio

# BPE 전후 비교
original = "l o w l o w l o w l o w e s t"
encoded = "low low low lowest"

ratio = calculate_compression_ratio(original, encoded)

print("\n=== 압축률 ===")
print(f"원본: '{original}' ({len(original.split())}개 토큰)")
print(f"BPE: '{encoded}' ({len(encoded.split())}개 토큰)")
print(f"압축률: {ratio:.2f}x")
```

### 실습 3: 엔트로피 비교
```python
from collections import Counter

def calculate_entropy(tokens):
    """토큰의 엔트로피 계산"""
    freq = Counter(tokens)
    total = len(tokens)
    probs = np.array([freq[t]/total for t in freq])
    return -np.sum(probs * np.log2(probs))

# 원본 vs BPE
original_tokens = "l o w l o w l o w l o w e s t".split()
bpe_tokens = "low low low lowest".split()

h_original = calculate_entropy(original_tokens)
h_bpe = calculate_entropy(bpe_tokens)

print("\n=== 엔트로피 비교 ===")
print(f"원본 엔트로피: {h_original:.4f} bits")
print(f"BPE 엔트로피: {h_bpe:.4f} bits")
print(f"감소: {h_original - h_bpe:.4f} bits")
print("→ BPE가 더 효율적!")
```

---

## ✍️ 손 계산 연습

### BPE 한 스텝
```
텍스트: "aa aa bb"
초기: a a   a a   b b

빈도 계산:
- (a, a): 2번
- (b, b): 1번

병합: (a, a) → aa
결과: aa   aa   b b
```

---

## 🔗 LLM 연결점

### 1. GPT/BERT의 토크나이저
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Tokenization is important!"
tokens = tokenizer.encode(text)

print(f"텍스트: {text}")
print(f"토큰 ID: {tokens}")
print(f"토큰: {tokenizer.convert_ids_to_tokens(tokens)}")

# BPE 기반!
```

### 2. 다국어 지원
```
BPE는 언어 독립적
- 바이트 기반
- 모든 언어에 적용 가능
- 미등록 단어(UNK) 최소화
```

---

## ✅ 체크포인트

- [ ] **BPE 알고리즘의 단계를 설명할 수 있나요?**
- [ ] **정보 이득과의 연결을 이해했나요?**
- [ ] **BPE의 장점을 아나요?**
- [ ] **실제 LLM에서의 활용을 이해했나요?**

---

## 🎓 핵심 요약

1. **BPE**: 빈번한 바이트 쌍 병합
2. **목표**: 압축, 엔트로피 감소
3. **장점**: 효율적, 언어 독립적
4. **LLM**: 거의 모든 모델이 BPE 사용

### 다음 학습
- **Day 40**: WordPiece와 SentencePiece

---

**수고하셨습니다!** 🎉

**BPE는 모든 LLM 토크나이저의 기초입니다!**
