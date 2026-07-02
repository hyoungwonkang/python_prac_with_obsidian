"""
저장된 가중치(spam_bert.pt / spam_klue.pt)를 로드해 test셋으로 평가하고
그 결과만 MLflow에 1회 기록한다 (재학습 없음).

- finetune_bert_spam.py 의 SpamDataset·evaluate·device 를 그대로 재사용 (학습 코드와 동일 경로 보장).
- 영어(bert-base-uncased / spam_bert.pt / ./test.csv)와
  한국어(klue/bert-base / spam_klue.pt / ./ko/test.csv)를
  같은 실험(bert-spam-classification) 안 별도 run으로 기록 → MLflow UI에서 나란히 비교.

실행: /Users/macrent/ml-env/bin/python eval_mlflow_spam.py
사전: MLflow 서버가 127.0.0.1:5000 에서 구동 중이어야 한다.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support,
)
import mlflow

# 학습 스크립트와 완전히 동일한 Dataset/평가 로직을 재사용
from finetune_bert_spam import SpamDataset, evaluate, device, BATCH_SIZE, MAX_LENGTH

HERE = Path(__file__).resolve().parent

# (run 이름, HF 모델, 가중치 파일, test.csv 경로)
CONFIGS = [
    ("en-bert-base-uncased", "bert-base-uncased", HERE / "spam_bert.pt",  HERE / "test.csv"),
    ("ko-klue-bert-base",    "klue/bert-base",    HERE / "spam_klue.pt",  HERE / "ko" / "test.csv"),
]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bert-spam-classification")


def eval_one(run_name, model_name, weights_path, test_csv):
    if not weights_path.exists():
        print(f"[skip] 가중치 없음: {weights_path}")
        return
    if not test_csv.exists():
        print(f"[skip] test.csv 없음: {test_csv}")
        return

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    test_ds = SpamDataset(test_csv, tokenizer)
    gold, pred = evaluate(model, DataLoader(test_ds, batch_size=BATCH_SIZE))

    acc = accuracy_score(gold, pred)
    cm = confusion_matrix(gold, pred)           # [[tn, fp], [fn, tp]] (0=ham, 1=spam)
    tn, fp, fn, tp = cm.ravel()
    # spam(=1) 기준 precision/recall/f1
    p, r, f1, _ = precision_recall_fscore_support(
        gold, pred, labels=[1], average="binary", zero_division=0)
    report = classification_report(
        gold, pred, target_names=["ham(0)", "spam(1)"], digits=4)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "eval-only")        # 재학습 아님, 평가만 기록
        mlflow.set_tag("weights", weights_path.name)
        mlflow.log_params({
            "model_name": model_name,
            "weights_file": weights_path.name,
            "test_csv": str(test_csv.relative_to(HERE)),
            "test_size": len(test_ds),
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "device": device,
        })
        mlflow.log_metrics({
            "test_accuracy": float(acc),
            "spam_precision": float(p),
            "spam_recall": float(r),
            "spam_f1": float(f1),
            "cm_tn": int(tn), "cm_fp": int(fp),
            "cm_fn": int(fn), "cm_tp": int(tp),
        })
        # 사람이 읽는 리포트는 artifact로 첨부
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "classification_report.txt"
            rp.write_text(
                f"model: {model_name}\nweights: {weights_path.name}\n"
                f"test: {test_csv} ({len(test_ds)}건)\n\n"
                f"accuracy: {acc:.4f}\n\nconfusion matrix [행=실제,열=예측] (0=ham,1=spam):\n{cm}\n\n{report}\n"
            )
            mlflow.log_artifact(str(rp))

    print(f"[logged] {run_name}: acc={acc:.4f} spam_f1={f1:.4f} (tn={tn} fp={fp} fn={fn} tp={tp})")


def main():
    for cfg in CONFIGS:
        eval_one(*cfg)
    print("\n완료 → MLflow UI: http://127.0.0.1:5000  (실험: bert-spam-classification)")


if __name__ == "__main__":
    main()
