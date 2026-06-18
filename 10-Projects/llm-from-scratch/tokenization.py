import urllib.request
import re

url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
file_path = "input.txt"
urllib.request.urlretrieve(url, file_path)

with open("input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("총 문자 개수:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("총 토큰 개수:", len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
print("어휘 크기:", len(all_words))
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """First Citizen:
        Before we proceed any further, hear me speak."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))