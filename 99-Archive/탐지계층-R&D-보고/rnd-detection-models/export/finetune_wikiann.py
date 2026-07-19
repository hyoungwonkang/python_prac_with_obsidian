"""
WikiAnn(ko) 토큰 분류 — 전이 사슬 A/B 실험 (NER 도메인 심화).

버트 본체 + 토큰분류층 구조는 finetune_ner.py와 동일. 다른 점:
  - 데이터: KLUE-NER(뉴스, 글자 단위, 13태그) → WikiAnn ko(위키, 어절 단위, 7태그)
    · 어절 단위라 "첫 subword만 라벨, 나머지 -100" 정렬 규칙이 실제로 발동
    · 자동 생성(silver) 라벨 — 품질 소음 있음 (gold인 KLUE와 대비)
  - 출발점 선택(INIT_FROM)으로 전이 사슬을 실측 비교:
      실험 A (INIT_FROM 없음)      : klue/bert-base 원본에서 출발  → wikiann_scratch.pt
      실험 B (INIT_FROM=ner_klue.pt): KLUE-NER 완제품의 본체에서 출발 → wikiann_from_ner.pt
    B의 분류층(13칸)은 태그 체계가 달라 이식 불가 → 본체만 이식, 분류층(7칸)은 난수 신규.

실행:
  WIKIANN_SUBSET=300 EPOCHS=1 ~/rnd-env/bin/python finetune_wikiann.py     # 스모크
  WIKIANN_SUBSET=8000 EPOCHS=2 ~/rnd-env/bin/python finetune_wikiann.py    # 실험 A
  WIKIANN_SUBSET=8000 EPOCHS=2 INIT_FROM=ner_klue.pt \
      ~/rnd-env/bin/python finetune_wikiann.py                             # 실험 B
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score
from datasets import load_dataset

from ner_dataset import align_labels          # subword -100 정렬 그대로 재사용
from finetune_ner import evaluate             # gold/pred 복원·채점 로직 재사용

MODEL_NAME = os.environ.get("MODEL_NAME", "klue/bert-base")
EPOCHS = int(os.environ.get("EPOCHS", 2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))
LR = float(os.environ.get("LR", 5e-5))
SUBSET = int(os.environ["WIKIANN_SUBSET"]) if os.environ.get("WIKIANN_SUBSET") else None
INIT_FROM = os.environ.get("INIT_FROM", "")   # ""=원본에서 / 경로=그 .pt의 본체에서 출발
SEED = 123

HERE = Path(__file__).resolve().parent
default_w = "wikiann_from_ner.pt" if INIT_FROM else "wikiann_scratch.pt"
WEIGHTS = Path(os.environ.get("WIKIANN_WEIGHTS", HERE / default_w))

torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


def build_tokenized(tokenizer):
    ds = load_dataset("unimelb-nlp/wikiann", "ko")
    label_list = ds["train"].features["ner_tags"].feature.names  # O, B/I-PER·ORG·LOC (7종)
    train, val = ds["train"], ds["validation"]
    if SUBSET:
        train = train.select(range(min(SUBSET, len(train))))
        val = val.select(range(min(max(SUBSET // 5, 1), len(val))))
    cols = train.column_names                                    # tokens·ner_tags·langs·spans
    train_tok = train.map(lambda x: align_labels(x, tokenizer, MAX_LENGTH),
                          batched=True, remove_columns=cols)
    val_tok = val.map(lambda x: align_labels(x, tokenizer, MAX_LENGTH),
                      batched=True, remove_columns=cols)
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return train_tok, val_tok, label_list


def main():
    init_desc = f"전이 2회차 (본체: {INIT_FROM})" if INIT_FROM else "전이 1회 (원본 klue/bert-base)"
    print(f"device: {device} | epochs: {EPOCHS} | subset: {SUBSET or 'full'} | {init_desc}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds, val_ds, label_list = build_tokenized(tokenizer)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    print(f"학습 {len(train_ds)} / 평가 {len(val_ds)} | 라벨 {len(label_list)}종: {label_list}")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    if INIT_FROM:
        # ner_klue.pt의 본체(1.1억)만 이식 — 분류층은 13칸(KLUE 체계)이라 7칸 새 머리와 모양 불일치 → 제외
        state = torch.load(HERE / INIT_FROM, map_location="cpu")
        body = {k: v for k, v in state.items() if not k.startswith("classifier")}
        model.load_state_dict(body, strict=False)
        print(f"본체 이식 완료: {len(body)}개 텐서 (분류층 {len(state)-len(body)}개 제외, 7칸 신규)")

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
        gold, pred = evaluate(model, DataLoader(val_ds, batch_size=BATCH_SIZE), id2label)
        print(f"[epoch {epoch}] train_loss {total/(len(train_ds)//BATCH_SIZE or 1):.4f} "
              f"| val_f1 {f1_score(gold, pred):.4f}")

    gold, pred = evaluate(model, DataLoader(val_ds, batch_size=BATCH_SIZE), id2label)
    print(f"\n=== WikiAnn(ko) 평가 ({len(val_ds)}건) — {init_desc} ===")
    print(f"entity F1: {f1_score(gold, pred):.4f}")
    print(classification_report(gold, pred, digits=4))

    torch.save(model.state_dict(), WEIGHTS)
    print(f"가중치 저장: {WEIGHTS.name}")


if __name__ == "__main__":
    main()
