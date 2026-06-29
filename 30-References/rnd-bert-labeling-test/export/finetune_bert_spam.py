"""
BERT 스팸 분류 파인튜닝 — BERT 모델 · 라벨링 · 테스트 실증.

코드를 위에서 아래로 읽으면 3주제가 순서대로 드러난다:
  ① 데이터·라벨 로드   → 라벨링
  ② Dataset/DataLoader → BERT 입력(토크나이즈+패딩)
  ③ 모델 로드          → BERT 모델 (BertForSequenceClassification)
  ④ 학습 루프          → BERT 파인튜닝
  ⑤ 테스트셋 평가      → 테스트 (accuracy·confusion matrix·precision/recall)

사전 준비: dataset_finetuning.py 를 먼저 실행해 train/validation/test.csv 를 만든다.
실행:      python finetune_bert_spam.py   (이 스크립트와 같은 폴더의 CSV를 읽음)
환경:      Apple Silicon(MPS)·GPU·CPU 모두 동작. TensorFlow 불필요.
"""
import os
os.environ["USE_TF"] = "0"        # Hugging Face가 TF/Flax를 임포트하지 않도록 차단
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── 설정 ──────────────────────────────────────────────────────────
# MODEL_NAME·DATA_DIR은 환경변수로 교체 가능 (기본=영어). 한국어는 MODEL_NAME=klue/bert-base.
MODEL_NAME = os.environ.get("MODEL_NAME", "bert-base-uncased")
MAX_LENGTH = 128          # SMS는 짧아 128이면 충분 (긴 것은 잘림)
BATCH_SIZE = 16
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = 2e-5                  # BERT 파인튜닝 표준 학습률
SEED = 123

# CSV는 이 스크립트와 같은 폴더에 있다고 가정 (dataset_finetuning.py 산출물). DATA_DIR로 교체 가능.
DATA_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).resolve().parent)

torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device} | epochs: {EPOCHS} | model: {MODEL_NAME} | data: {DATA_DIR}")


# ── ② Dataset: SMS 1건을 토크나이즈+패딩해 모델 입력으로 ────────────
class SpamDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["Text"]),
            padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["Label"]), dtype=torch.long),
        }


def evaluate(model, loader):
    """주어진 loader에 대해 예측 → (정답, 예측) 반환 (평가 공통 함수)."""
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            golds.extend(batch["label"].tolist())
    return golds, preds


def main():
    # ── ① 데이터·라벨 로드 [라벨링] ────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_ds = SpamDataset(DATA_DIR / "train.csv", tokenizer)
    val_ds = SpamDataset(DATA_DIR / "validation.csv", tokenizer)
    test_ds = SpamDataset(DATA_DIR / "test.csv", tokenizer)
    print(f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ── ③ 모델 [BERT 모델] ─────────────────────────────────────────
    # [CLS] 벡터 위에 분류층(num_labels=2)을 얹는 텍스트 분류 패턴
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # ── ④ 학습 루프 [BERT 파인튜닝] ────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
        avg = total / len(train_loader)
        val_gold, val_pred = evaluate(model, val_loader)
        val_acc = accuracy_score(val_gold, val_pred)
        print(f"[epoch {epoch}] train_loss {avg:.4f} | val_acc {val_acc:.4f}")

    # ── ⑤ 테스트셋 최종 평가 [테스트] ──────────────────────────────
    # 학습에 한 번도 쓰지 않은 test셋(300건)으로 실전 성능을 딱 한 번 채점
    gold, pred = evaluate(model, test_loader)
    acc = accuracy_score(gold, pred)
    cm = confusion_matrix(gold, pred)
    print("\n=== TEST 결과 (test.csv 300건, 학습 미사용) ===")
    print(f"accuracy: {acc:.4f}")
    print("confusion matrix [행=실제, 열=예측] (0=ham,1=spam):")
    print(cm)
    print("\nclassification report:")
    print(classification_report(gold, pred, target_names=["ham(0)", "spam(1)"], digits=4))

    # 학습된 가중치 저장 (WEIGHTS 환경변수로 파일명 교체 — 모델별 분리해 덮어쓰기 방지)
    weights_out = os.environ.get("WEIGHTS", "spam_bert.pt")
    torch.save(model.state_dict(), weights_out)
    print(f"\n저장 완료: {weights_out}")


if __name__ == "__main__":
    main()
