"""
NER 추론/데모 — 학습된 가중치(ner_klue.pt)로 한국어 문장에서 개체명을 추출한다.

플랫폼의 "NER Agent(인명·기관·지명·날짜·수량·시간 추출)"에 대응하는 최소 추론기.
학습과 동일하게 문자 단위(is_split_into_words)로 토큰화한 뒤, 각 문자의 첫 subword
예측만 모아 BIO 태그를 엔티티 구간으로 병합한다.

실행:
  python predict_ner.py            # 샘플 문장 + 대화형
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ner_dataset import load_klue_ner

MODEL_NAME = os.environ.get("MODEL_NAME", "klue/bert-base")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))
HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("WEIGHTS", HERE / "ner_klue.pt"))

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

SAMPLES = [
    "홍길동은 2024년 3월에 서울시청 앞에서 김철수를 만났다.",
    "삼성전자는 수원에 위치한 대기업이다.",
    "내일 오후 3시에 부산에서 회의가 열린다.",
]


def load():
    _, label_list, label2id, id2label = load_klue_ner()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device).eval()
    return tokenizer, model, id2label


def tag_sentence(text, tokenizer, model, id2label):
    """문자 단위 예측 → BIO를 (개체 텍스트, 유형) 구간으로 병합."""
    chars = list(text)
    enc = tokenizer(chars, is_split_into_words=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
    preds = logits.argmax(dim=-1)[0].cpu().tolist()

    # 각 문자(첫 subword)의 태그
    char_tags, prev = [], None
    for wid, p in zip(word_ids, preds):
        if wid is None or wid == prev:
            prev = wid
            continue
        char_tags.append((chars[wid], id2label[p]))
        prev = wid

    # BIO → 구간 병합
    entities, cur, cur_type = [], "", None
    for ch, tag in char_tags:
        if tag.startswith("B-"):
            if cur:
                entities.append((cur, cur_type))
            cur, cur_type = ch, tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur += ch
        else:
            if cur:
                entities.append((cur, cur_type))
            cur, cur_type = "", None
    if cur:
        entities.append((cur, cur_type))
    return entities


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"가중치 없음: {WEIGHTS} — 먼저 finetune_ner.py를 실행하세요.")
    tokenizer, model, id2label = load()

    print("=== 샘플 문장 개체명 추출 ===")
    for s in SAMPLES:
        ents = tag_sentence(s, tokenizer, model, id2label)
        print(f"\n> {s}")
        for txt, typ in ents:
            print(f"   [{typ}] {txt}")

    print("\n=== 대화형 (빈 줄 입력 시 종료) ===")
    while True:
        try:
            text = input("문장> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        for txt, typ in tag_sentence(text, tokenizer, model, id2label):
            print(f"   [{typ}] {txt}")


if __name__ == "__main__":
    main()
