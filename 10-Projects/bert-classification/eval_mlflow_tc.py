"""
저장된 가중치(bert_klue_tc.pt)를 로드해 KLUE-TC validation으로 평가하고
그 결과만 통합 MLflow 서버(127.0.0.1:5000)에 1회 기록한다 (재학습 없음).

- eval_mlflow_spam.py / eval_mlflow_ner.py와 같은 관례 — 학습 기록은 프로젝트
  로컬(sqlite)에, 주간보고용 최종 지표는 통합 서버에.
- finetune_klue_tc.py의 TcDataset·evaluate·load_klue_tc를 그대로 재사용.

실행 (보고 수치와 같은 평가셋 2,000건 기준):
  TC_SUBSET=10000 ~/rnd-env/bin/python eval_mlflow_tc.py
사전: MLflow 서버가 127.0.0.1:5000에서 구동 중 + bert_klue_tc.pt 존재.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score
import mlflow

from finetune_klue_tc import (
    TcDataset, evaluate, load_klue_tc, device, MODEL_NAME, MAX_LENGTH, BATCH_SIZE)

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("WEIGHTS", HERE / "bert_klue_tc.pt"))

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bert-klue-tc")


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"가중치 없음: {WEIGHTS} — 먼저 finetune_klue_tc.py 실행.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    _, val_raw, label_names = load_klue_tc()
    val_loader = DataLoader(TcDataset(val_raw, tokenizer), batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_names))
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device)

    gold, pred = evaluate(model, val_loader)
    acc = accuracy_score(gold, pred)
    macro = f1_score(gold, pred, average="macro")
    report = classification_report(gold, pred, target_names=label_names, digits=4)

    with mlflow.start_run(run_name="klue-bert-base-tc-7class"):
        mlflow.set_tag("stage", "eval-only")
        mlflow.set_tag("weights", WEIGHTS.name)
        mlflow.log_params({
            "모델": MODEL_NAME,
            "가중치_파일": WEIGHTS.name,
            "클래스_수": len(label_names),
            "평가_건수": len(val_raw),
            "최대_길이": MAX_LENGTH,
            "배치_크기": BATCH_SIZE,
            "장치": device,
        })
        mlflow.log_metrics({
            "정확도": float(acc),
            "매크로_F1": float(macro),
        })
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "classification_report.txt"
            rp.write_text(
                f"model: {MODEL_NAME}\nweights: {WEIGHTS.name}\n"
                f"eval(KLUE-TC validation): {len(val_raw)}건\n\n"
                f"accuracy: {acc:.4f} | macro F1: {macro:.4f}\n\n{report}\n")
            mlflow.log_artifact(str(rp))

    print(f"[logged] klue-bert-base-tc-7class: acc={acc:.4f} macro_f1={macro:.4f}")
    print("완료 → MLflow UI: http://127.0.0.1:5000 (실험: bert-klue-tc)")


if __name__ == "__main__":
    main()
