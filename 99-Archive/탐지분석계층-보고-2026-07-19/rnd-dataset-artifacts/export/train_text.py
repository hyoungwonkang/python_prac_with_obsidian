"""
텍스트 분류 범용 학습기 — `text,label` CSV만 주면 어떤 분류 과제든 같은 방식으로 학습한다.

확장성 규약(입력 계약):
  - DATA = CSV 파일 1개  → 70:10:20 자동 분할 (seed 123, 기존 스팸 R&D와 동일)
  - DATA = 디렉터리      → train.csv / validation.csv / test.csv 를 그대로 사용
  - 열 이름은 자동 감지: 'text' 포함 열 → 입력, 'label' 포함 열 → 정답 (대소문자 무관)
  - 라벨은 문자열/숫자 모두 허용 → 정렬 후 자동 번호 부여 (클래스 수 자동 인식)

산출물 규약(출력 계약) — artifacts/<NAME>/ 아래 3종 세트:
  - model.pt        학습된 가중치 (state_dict)
  - label_map.json  숫자↔클래스명 매핑 (이게 없으면 가중치를 재사용 못함)
  - meta.json       베이스모델·파라미터·지표·MLflow run id (재현 정보)

실행 예:
  DATA=../../rnd-bert-labeling-test/export/ko LIMIT=300 EPOCHS=1 NAME=ko-spam-smoke \\
    python train_text.py
"""
import json
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent

DATA = Path(os.environ.get("DATA", HERE / "../../rnd-bert-labeling-test/export/ko")).resolve()
BASE = os.environ.get("BASE", "klue/bert-base")
EPOCHS = int(os.environ.get("EPOCHS", 1))
BATCH = int(os.environ.get("BATCH", 16))
LR = float(os.environ.get("LR", 2e-5))
MAX_LEN = int(os.environ.get("MAX_LEN", 128))
LIMIT = int(os.environ.get("LIMIT", 0))          # 0 = 전체, N = 분할별 상한 (스모크용)
NAME = os.environ.get("NAME", DATA.stem)          # 산출물 이름
EXPERIMENT = os.environ.get("EXPERIMENT", "텍스트분류-범용")
OUT_DIR = HERE / "artifacts" / NAME

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))  # 맥=MPS / 윈도우 GPU=CUDA / 그 외=CPU


def find_columns(df):
    """열 이름에서 text/label 열을 자동 감지 (대소문자 무관)."""
    text_col = label_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "text" in lc and text_col is None:
            text_col = c
        elif "label" in lc and label_col is None:
            label_col = c
    if text_col is None or label_col is None:
        raise SystemExit(f"❌ text/label 열을 찾지 못함 — 실제 열: {list(df.columns)}")
    return text_col, label_col


def load_splits():
    """DATA가 디렉터리면 3분할 CSV, 파일이면 자동 70:10:20 분할."""
    if DATA.is_dir():
        splits = {}
        for part in ("train", "validation", "test"):
            p = DATA / f"{part}.csv"
            if not p.exists():
                raise SystemExit(f"❌ {p} 없음 — 디렉터리 규약: train/validation/test.csv")
            splits[part] = pd.read_csv(p)
        return splits["train"], splits["validation"], splits["test"]
    df = pd.read_csv(DATA)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    t_end = int(len(df) * 0.7)
    v_end = t_end + int(len(df) * 0.1)
    return df[:t_end], df[t_end:v_end], df[v_end:]


class TextDataset(Dataset):
    def __init__(self, texts, label_ids, tokenizer):
        enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                        padding="max_length", return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i]}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        preds += logits.argmax(-1).cpu().tolist()
        golds += batch["labels"].cpu().tolist()
    return preds, golds


def main():
    train_df, val_df, test_df = load_splits()
    text_col, label_col = find_columns(train_df)
    if LIMIT:
        train_df, val_df, test_df = (d.head(LIMIT) for d in (train_df, val_df, test_df))

    # 라벨 체계 자동 구축 — 문자열이든 숫자든 정렬해 고정 번호 부여
    classes = sorted({str(v) for d in (train_df, val_df, test_df) for v in d[label_col]})
    label2id = {c: i for i, c in enumerate(classes)}
    print(f"데이터: {DATA}")
    print(f"열 감지: text={text_col!r}, label={label_col!r} / 클래스 {len(classes)}개: {classes}")
    print(f"분할: train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")
    # 계약 위반을 조기에 친절히 — 빈 분할이 토크나이저까지 흘러가면 IndexError로만 죽는다
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise SystemExit("❌ 데이터가 너무 적습니다 — 70:10:20 분할 후 빈 몫이 생겼습니다. "
                         "단독 학습에는 최소 수십 건이 필요합니다. 소량 검수분은 기존 train에 편입해 학습하세요.")
    if len(classes) < 2:
        raise SystemExit(f"❌ 클래스가 {len(classes)}개뿐입니다 {classes} — 분류 학습에는 두 클래스 이상 필요 "
                         "(검수 데이터라면 정상·스팸 양쪽을 모두 모아야 합니다).")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(classes)).to(DEVICE)

    def make_loader(df, shuffle=False):
        ds = TextDataset(df[text_col].astype(str).tolist(),
                         [label2id[str(v)] for v in df[label_col]], tokenizer)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # MLflow 3.x: 파일 저장소는 유지보수 모드 → 실행 위치와 무관하게 sqlite로 고정
    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.log_params({"베이스모델": BASE, "에폭": EPOCHS, "배치": BATCH,
                           "학습률": LR, "최대길이": MAX_LEN, "클래스수": len(classes),
                           "데이터": str(DATA), "학습샘플수": len(train_df)})
        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                loss = model(**batch).loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                          f"loss {loss.item():.4f}")
            preds, golds = evaluate(model, val_loader)
            val_acc = accuracy_score(golds, preds)
            mlflow.log_metric("검증정확도", val_acc, step=epoch)
            print(f"epoch {epoch}: 검증정확도 {val_acc:.4f}")

        preds, golds = evaluate(model, test_loader)
        test_acc = accuracy_score(golds, preds)
        test_f1 = f1_score(golds, preds, average="macro")
        mlflow.log_metrics({"테스트정확도": test_acc, "테스트매크로F1": test_f1})
        print(f"테스트: 정확도 {test_acc:.4f} / 매크로F1 {test_f1:.4f} "
              f"({time.time()-t0:.0f}초)")

        # ---- 산출물 3종 세트 저장 ----
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), OUT_DIR / "model.pt")
        (OUT_DIR / "label_map.json").write_text(
            json.dumps({"label2id": label2id,
                        "id2label": {str(i): c for c, i in label2id.items()}},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT_DIR / "meta.json").write_text(
            json.dumps({"name": NAME, "task": "text-classification",
                        "base_model": BASE, "max_len": MAX_LEN,
                        "num_labels": len(classes), "data": str(DATA),
                        "metrics": {"테스트정확도": round(test_acc, 4),
                                    "테스트매크로F1": round(test_f1, 4)},
                        "mlflow_run_id": run.info.run_id},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"산출물 저장 → {OUT_DIR}/ (model.pt + label_map.json + meta.json)")


if __name__ == "__main__":
    main()
