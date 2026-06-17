import urllib.request

url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
file_path = "input.txt"
urllib.request.urlretrieve(url, file_path)

with open("input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("총 문자 개수:", len(raw_text))
print(raw_text[:99])