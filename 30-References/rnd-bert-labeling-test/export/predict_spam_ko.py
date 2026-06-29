"""
한국어 실시간 스팸 예측 시연 — klue/bert-base + spam_klue.pt 로 한글 문장을 즉석 분류.

사전 준비:
  WEIGHTS=spam_klue.pt MODEL_NAME=klue/bert-base DATA_DIR=./ko python finetune_bert_spam.py
  → spam_klue.pt 생성
실행:
  python predict_spam_ko.py
  → ① 한국어 샘플 일괄 판정  ② 대화형(문장 입력 → 스팸/정상 + 확신도, 빈 줄/Ctrl-D 종료)
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, logging

logging.set_verbosity_error()

MODEL_NAME = "klue/bert-base"        # 한국어 사전학습 BERT
MAX_LENGTH = 128
WEIGHTS = Path(__file__).resolve().parent / "spam_klue.pt"

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.to(device)
model.eval()
print(f"모델 로드 완료: {WEIGHTS.name} ({MODEL_NAME}) | device: {device}\n")

LABELS = {0: "✅ 정상(ham)", 1: "🚨 스팸(spam)"}


def predict(text):
    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0]
    pred = int(probs.argmax())
    return LABELS[pred], probs[pred].item()


def show(text):
    label, conf = predict(text)
    print(f"  입력> {text}")
    print(f"   → {label} (확신도 {conf*100:.1f}%)\n")


# ── ① 한국어 샘플 일괄 판정 ───────────────────────────────────────
SAMPLES = [
    "[Web발신] 축하합니다! 100만원 경품에 당첨되셨습니다. 지금 클릭 http://bit.ly/win",
    "오늘 저녁 7시에 만나는 거 맞지?",
    "무료 대출 가능! 무방문 당일 입금, 지금 전화주세요 010-1234-5678",
    "엄마 마트에서 우유 좀 사다 줄 수 있어?",
    "지금 가입하면 무료 상품권 즉시 지급! 링크 클릭하세요",
    "회의 자료 메일로 보냈어요, 확인 부탁드립니다",
]
print("=== 한국어 샘플 문장 일괄 판정 ===")
for s in SAMPLES:
    show(s)

# ── ② 대화형 판정 ─────────────────────────────────────────────────
print("=== 대화형 (문장 입력 후 Enter / 빈 줄·Ctrl-D 로 종료) ===")
while True:
    try:
        text = input("문장 입력> ").strip()
    except EOFError:
        print("\n종료합니다.")
        break
    if not text:
        print("종료합니다.")
        break
    show(text)
