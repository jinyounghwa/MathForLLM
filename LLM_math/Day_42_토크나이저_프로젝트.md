# Day 42: 토크나이저 프로젝트 (2시간)

## 📚 학습 목표
- 완전한 BPE 토크나이저 구현하기
- 인코딩/디코딩 기능 만들기
- 압축률 평가하기

---

## 🎯 프로젝트
**"나만의 BPE 토크나이저 만들기"**

---

## 💻 최종 프로젝트 코드

```python
import re
from collections import Counter, defaultdict

class BPETokenizer:
    """완전한 BPE 토크나이저"""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.vocab = set()

    def get_stats(self, words):
        """바이트 쌍 빈도"""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_pair(self, pair, words):
        """쌍 병합"""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        pattern = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')

        for word in words:
            new_word = p.sub(replacement, word)
            new_words[new_word] = words[word]
        return new_words

    def train(self, corpus):
        """학습"""
        # 단어 빈도
        words = corpus.lower().split()
        word_freqs = Counter(words)

        # 문자로 분리
        vocab_words = {' '.join(word): freq
                      for word, freq in word_freqs.items()}

        # 초기 어휘 (문자)
        for word in vocab_words:
            self.vocab.update(word.split())

        print(f"=== BPE 학습 ===\n")
        print(f"초기 어휘 크기: {len(self.vocab)}")

        # BPE 학습
        for i in range(self.vocab_size - len(self.vocab)):
            pairs = self.get_stats(vocab_words)

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab_words = self.merge_pair(best_pair, vocab_words)

            self.bpe_codes[best_pair] = i
            self.vocab.add(''.join(best_pair))

            if i % 10 == 0 or i < 5:
                print(f"  {i+1}. {best_pair[0]} + {best_pair[1]} → "
                      f"{''.join(best_pair)} (빈도: {pairs[best_pair]})")

        print(f"\n최종 어휘 크기: {len(self.vocab)}")

    def encode(self, text):
        """텍스트 → 토큰"""
        words = text.lower().split()
        encoded = []

        for word in words:
            word_tokens = ' '.join(word)

            # BPE 적용
            while True:
                pairs = [(word_tokens.split()[i], word_tokens.split()[i+1])
                        for i in range(len(word_tokens.split())-1)]

                if not pairs:
                    break

                # 학습된 순서대로 병합
                min_pair = min(pairs,
                              key=lambda p: self.bpe_codes.get(p, float('inf')))

                if min_pair not in self.bpe_codes:
                    break

                first, second = min_pair
                new_word = []
                i = 0
                symbols = word_tokens.split()

                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(symbols[i])
                        i += 1

                word_tokens = ' '.join(new_word)

            encoded.extend(word_tokens.split())

        return encoded

    def decode(self, tokens):
        """토큰 → 텍스트"""
        return ' '.join(''.join(tokens).split())

# 사용 예
corpus = """
hello hello hello world
hello world
world of warcraft
"""

# 학습
tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(corpus)

# 테스트
test_text = "hello world"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"\n=== 테스트 ===")
print(f"원본: {test_text}")
print(f"토큰: {tokens}")
print(f"디코딩: {decoded}")

# 압축률
original_chars = len(test_text.replace(' ', ''))
num_tokens = len(tokens)
compression = original_chars / num_tokens

print(f"\n압축률: {compression:.2f}x")
```

---

## 🎓 프로젝트 완료!

**당신은 이제 토크나이저를 만들 수 있습니다!**

### 다음 단계
- **Day 43**: Scaled Dot-Product Attention
  - LLM의 핵심 메커니즘!

---

**축하합니다!** 🎉
