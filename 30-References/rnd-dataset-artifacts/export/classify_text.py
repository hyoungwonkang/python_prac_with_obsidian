"""
산출물 재사용 분류기 — artifacts/<이름>/ 3종 세트(model.pt + label_map.json + meta.json)를
불러와 새 텍스트를 분류한다. "학습은 한 번, 분류는 계속"의 실행 증명.

실행 예:
  ARTIFACT=artifacts/ko-spam-smoke ~/rnd-env/bin/python classify_text.py "무료 쿠폰 당첨!"
  (인자 없으면 내장 데모 문장 사용)
"""
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ARTIFACT = Path(os.environ.get("ARTIFACT", HERE / "artifacts/ko-spam-smoke"))
if not ARTIFACT.is_absolute():
    ARTIFACT = HERE / ARTIFACT

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DEMO_TEXTS = [
    "무료 쿠폰이 당첨되셨습니다! 지금 바로 클릭하세요",
    "내일 회의 자료 공유드립니다. 확인 부탁드려요.",
]


def main():
    meta = json.loads((ARTIFACT / "meta.json").read_text(encoding="utf-8"))
    label_map = json.loads((ARTIFACT / "label_map.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    print(f"산출물: {ARTIFACT.name} (베이스 {meta['base_model']}, "
          f"클래스 {meta['num_labels']}개, 지표 {meta['metrics']})")

    tokenizer = AutoTokenizer.from_pretrained(meta["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=meta["num_labels"])
    model.load_state_dict(torch.load(ARTIFACT / "model.pt", map_location="cpu"))
    model.to(DEVICE).eval()

    texts = sys.argv[1:] or DEMO_TEXTS
    enc = tokenizer(texts, truncation=True, max_length=meta["max_len"],
                    padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    for text, p in zip(texts, probs):
        top = int(p.argmax())
        print(f"  [{id2label[top]}] ({p[top]:.2%}) {text}")


if __name__ == "__main__":
    main()
