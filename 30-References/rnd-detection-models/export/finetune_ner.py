"""
NER 학습·평가 — klue/bert-base + BertForTokenClassification (KLUE-NER).

R&D 3주제(모델·라벨링·테스트)의 NER 판, 위에서 아래로:
  ① 데이터/라벨 로드    → ner_dataset.build_tokenized (subword -100 정렬)
  ② 모델                → BertForTokenClassification (BERT 본체 + 토큰 분류층)
  ③ 학습 루프           → AdamW
  ④ 테스트(평가)        → seqeval 엔티티 단위 정밀도/재현율/F1

환경변수로 파라미터화 (finetune_bert_spam.py 스타일):
  MODEL_NAME(기본 klue/bert-base) · EPOCHS(2) · BATCH_SIZE(16) · MAX_LENGTH(128)
  · LR(5e-5) · NER_SUBSET(정수면 그만큼만 — 스모크 테스트용)

실행:
  python finetune_ner.py                 # 전체
  NER_SUBSET=800 EPOCHS=1 python finetune_ner.py   # 빠른 스모크
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

from ner_dataset import build_tokenized

MODEL_NAME = os.environ.get("MODEL_NAME", "klue/bert-base")
EPOCHS = int(os.environ.get("EPOCHS", 2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))
LR = float(os.environ.get("LR", 5e-5))
SUBSET = int(os.environ["NER_SUBSET"]) if os.environ.get("NER_SUBSET") else None
SEED = 123

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("WEIGHTS", HERE / "ner_klue.pt"))

torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, id2label):
    """예측 → -100 위치 제외하고 태그 문자열 시퀀스로 복원 → seqeval 입력."""
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            preds = logits.argmax(dim=-1).cpu()
            for p_row, l_row in zip(preds, labels):
                t_seq, p_seq = [], []
                for p, l in zip(p_row.tolist(), l_row.tolist()):
                    if l == -100:            # subword 이어짐/특수토큰/패딩 → 평가 제외
                        continue
                    t_seq.append(id2label[l])
                    p_seq.append(id2label[p])
                true_tags.append(t_seq)
                pred_tags.append(p_seq)
    return true_tags, pred_tags


def build_model(num_labels, id2label, label2id):
    # Auto는 자동 선택기 — klue/bert-base(model_type=bert)이므로 실물은 BertForTokenClassification이 생성됨
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id)


def main():
    print(f"device: {device} | model: {MODEL_NAME} | epochs: {EPOCHS} "
          f"| subset: {SUBSET or 'full'}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds, val_ds, label_list, label2id, id2label = build_tokenized(
        tokenizer, MAX_LENGTH, SUBSET)
    print(f"학습 {len(train_ds)} / 평가 {len(val_ds)} | 라벨 {len(label_list)}종")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model(len(label_list), id2label, label2id).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
        gold, pred = evaluate(model, val_loader, id2label)
        print(f"[epoch {epoch}] train_loss {total/len(train_loader):.4f} "
              f"| val_f1 {f1_score(gold, pred):.4f}")

    # 최종 평가 (엔티티 단위)
    gold, pred = evaluate(model, val_loader, id2label)
    print(f"\n=== NER 평가 (KLUE-NER validation {len(val_ds)}건) ===")
    print(f"entity F1: {f1_score(gold, pred):.4f}")
    print(classification_report(gold, pred, digits=4))

    torch.save(model.state_dict(), WEIGHTS)
    print(f"가중치 저장: {WEIGHTS.name}")


if __name__ == "__main__":
    main()
