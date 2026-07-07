"""
Phase 1 — KLUE-TC(ynat) 뉴스 주제 7클래스 분류 (klue/bert-base 파인튜닝).

finetune_bert_spam.py(2클래스) 골격 재사용 — 핵심 변경은 두 가지뿐:
  ① num_labels 2 → 7 (분류층 칸 수)
  ② 데이터: CSV 직접 분할 → HF datasets의 KLUE-TC (train/validation 제공,
     공식 test는 라벨 비공개라 validation을 평가셋으로 사용 — KLUE-NER과 동일)

평가: 다중 클래스라 accuracy 외에 **클래스별 P/R/F1 + macro 평균** 확인.
MLflow: 한글 키 관례 (실험 bert-klue-tc, file store).

환경변수: MODEL_NAME(klue/bert-base) · EPOCHS(2) · BATCH_SIZE(32) · MAX_LENGTH(64)
  · LR(2e-5) · TC_SUBSET(정수면 그만큼만 — 스모크용) · WEIGHTS(bert_klue_tc.pt)
  ※ MAX_LENGTH=64: 입력이 뉴스 '제목'(짧음)이라 128 불필요 — 계산 절약.

실행:
  TC_SUBSET=500 EPOCHS=1 ~/rnd-env/bin/python finetune_klue_tc.py   # 스모크
  TC_SUBSET=10000 ~/rnd-env/bin/python finetune_klue_tc.py          # 본 실행
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score
from datasets import load_dataset

MODEL_NAME = os.environ.get("MODEL_NAME", "klue/bert-base")
EPOCHS = int(os.environ.get("EPOCHS", 2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 64))
LR = float(os.environ.get("LR", 2e-5))
SUBSET = int(os.environ["TC_SUBSET"]) if os.environ.get("TC_SUBSET") else None
SEED = 123

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("WEIGHTS", HERE / "bert_klue_tc.pt"))

torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


# ① 데이터 — KLUE-TC(ynat): 뉴스 제목 + 주제 라벨(0~6)
def load_klue_tc():
    ds = load_dataset("klue/klue", "ynat")
    label_names = ds["train"].features["label"].names   # 7클래스 이름
    train, val = ds["train"], ds["validation"]
    if SUBSET:
        train = train.select(range(min(SUBSET, len(train))))
        val = val.select(range(min(max(SUBSET // 5, 1), len(val))))
    return train, val, label_names


# ② Dataset: 제목 1건을 토크나이즈 + 패딩 (SpamDataset과 동일 구조)
class TcDataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data = hf_split
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        enc = self.tokenizer(row["title"], padding="max_length",
                             truncation=True, max_length=MAX_LENGTH,
                             return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(int(row["label"]), dtype=torch.long)}


def evaluate(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            golds.extend(batch["label"].tolist())
    return golds, preds


def main():
    print(f"device: {device} | model: {MODEL_NAME} | epochs: {EPOCHS} "
          f"| subset: {SUBSET or 'full'}")
    train_raw, val_raw, label_names = load_klue_tc()
    print(f"학습 {len(train_raw)} / 평가 {len(val_raw)} | 클래스 {len(label_names)}종: {label_names}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_loader = DataLoader(TcDataset(train_raw, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TcDataset(val_raw, tokenizer), batch_size=BATCH_SIZE)

    # ③ 모델 — 스팸과 유일한 구조 차이: num_labels 2 → 7
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_names)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # MLflow (한글 키 관례) — sqlite 백엔드, 프로젝트 폴더에 고정
    # ※ mlflow 3.14부터 파일 스토어(file:./mlruns)는 유지보수 모드로 예외 발생 → sqlite 권장 방식 사용
    #   (ml-env의 3.1.4에선 file store가 되던 것 — 의존성 버전 차이 실사례)
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    mlflow.set_experiment("bert-klue-tc")

    start = time.time()
    with mlflow.start_run(run_name=f"7class-{len(train_raw)}건"):
        mlflow.log_params({
            "모델": MODEL_NAME, "클래스_수": len(label_names),
            "학습률": LR, "에폭_수": EPOCHS, "배치_크기": BATCH_SIZE,
            "최대_길이": MAX_LENGTH,
            "학습_데이터_수": len(train_raw), "평가_건수": len(val_raw),
            "장치": device,
        })

        # ④ 학습 루프 (spam과 동일)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["label"].to(device))
                out.loss.backward()
                optimizer.step()
                total += out.loss.item()
            gold, pred = evaluate(model, val_loader)
            acc = accuracy_score(gold, pred)
            print(f"[epoch {epoch}] train_loss {total/len(train_loader):.4f} "
                  f"| val_acc {acc:.4f}")
            mlflow.log_metric("훈련_손실", total / len(train_loader), step=epoch)
            mlflow.log_metric("검증_정확도", acc, step=epoch)

        # ⑤ 최종 평가 — 다중 클래스: 클래스별 P/R/F1 + macro
        gold, pred = evaluate(model, val_loader)
        acc = accuracy_score(gold, pred)
        macro = f1_score(gold, pred, average="macro")
        minutes = (time.time() - start) / 60
        print(f"\n=== 평가 (KLUE-TC validation {len(val_raw)}건) ===")
        print(f"accuracy: {acc:.4f} | macro F1: {macro:.4f} | 소요 {minutes:.1f}분")
        print(classification_report(gold, pred, target_names=label_names, digits=4))
        mlflow.log_metrics({"정확도": acc, "매크로_F1": macro, "훈련_시간_분": minutes})

    torch.save(model.state_dict(), WEIGHTS)
    print(f"가중치 저장: {WEIGHTS.name}")


if __name__ == "__main__":
    main()
