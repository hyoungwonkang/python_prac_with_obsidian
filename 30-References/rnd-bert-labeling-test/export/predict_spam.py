"""
실시간 스팸 예측 시연 — 학습된 spam_bert.pt 를 불러와 문장을 즉석 분류한다.

사전 준비: finetune_bert_spam.py 를 한 번 실행해 spam_bert.pt 를 만들어 둔다.
실행:      python predict_spam.py
           → ① 샘플 문장 일괄 판정 출력
           → ② 대화형: 문장을 입력하면 스팸/정상 + 확신도 출력 (빈 줄 또는 Ctrl-D로 종료)
"""
import os
import warnings
warnings.filterwarnings("ignore")   # urllib3 NotOpenSSLWarning 등 무해 경고 숨김 (시연 화면 정리)
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, logging

logging.set_verbosity_error()   # 시연 중 "should TRAIN this model" 오해성 경고 숨김
# (분류층은 아래 .pt 로드로 즉시 덮어쓰므로 그 경고는 무의미)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
WEIGHTS = Path(__file__).resolve().parent / "spam_bert.pt"

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

# ① 빈 모델 구조 만들고 → ② 학습된 가중치(.pt) 끼우기
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.to(device)
model.eval()
print(f"모델 로드 완료: {WEIGHTS.name} | device: {device}\n")

LABELS = {0: "✅ 정상(ham)", 1: "🚨 스팸(spam)"}


def predict(text):
    """문장 하나 → (라벨, 확신도). 확신도 = 예측 클래스의 확률."""
    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0]   # [P(정상), P(스팸)]
    pred = int(probs.argmax())
    return LABELS[pred], probs[pred].item()


def show(text):
    label, conf = predict(text)
    print(f"  입력> {text}")
    print(f"   → {label} (확신도 {conf*100:.1f}%)\n")


# ── ① 샘플 일괄 판정 ──────────────────────────────────────────────
SAMPLES = [
    "Free entry! Win a prize now, click here to claim",
    "Hey, are we still meeting at 7 tonight?",
    "URGENT! You have won £1000 cash. Call 09061701461 now",
    "Can you pick up some milk on your way home?",
    "Congratulations! Claim your free ringtone, text WIN to 80086",
    "I'll be 10 minutes late, sorry",
]
print("=== 샘플 문장 일괄 판정 ===")
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
