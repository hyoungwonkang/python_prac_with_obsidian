"""
저장된 NER 가중치(ner_klue.pt)를 로드해 KLUE-NER validation으로 평가하고
그 결과만 MLflow에 1회 기록한다 (재학습 없음).

- finetune_ner.py 의 evaluate·build_model·device 등을 그대로 재사용.
- 지표는 seqeval 엔티티 단위 정밀도/재현율/F1. MLflow 키는 한글 관례 유지
  (detection-ai-study 공통 규칙).

실행: ~/rnd-env/bin/python eval_mlflow_ner.py
사전: MLflow 서버가 127.0.0.1:5000 에서 구동 중 + ner_klue.pt 존재.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import (
    classification_report, f1_score, precision_score, recall_score)
import mlflow

from ner_dataset import build_tokenized
from finetune_ner import (
    evaluate, build_model, device, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, WEIGHTS)

SUBSET = int(os.environ["NER_SUBSET"]) if os.environ.get("NER_SUBSET") else None

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ner-klue")


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"가중치 없음: {WEIGHTS} — 먼저 finetune_ner.py 실행.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    _, val_ds, label_list, label2id, id2label = build_tokenized(
        tokenizer, MAX_LENGTH, SUBSET)

    model = build_model(len(label_list), id2label, label2id)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device)

    gold, pred = evaluate(model, DataLoader(val_ds, batch_size=BATCH_SIZE), id2label)
    f1 = f1_score(gold, pred)
    p = precision_score(gold, pred)
    r = recall_score(gold, pred)
    report = classification_report(gold, pred, digits=4)

    with mlflow.start_run(run_name="klue-bert-base-ner"):
        mlflow.set_tag("stage", "eval-only")
        mlflow.set_tag("weights", WEIGHTS.name)
        mlflow.log_params({
            "모델": MODEL_NAME,
            "가중치_파일": WEIGHTS.name,
            "평가_건수": len(val_ds),
            "최대_길이": MAX_LENGTH,
            "배치_크기": BATCH_SIZE,
            "라벨_수": len(label_list),
            "장치": device,
        })
        mlflow.log_metrics({
            "엔티티_F1": float(f1),
            "엔티티_정밀도": float(p),
            "엔티티_재현율": float(r),
        })
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "ner_report.txt"
            rp.write_text(
                f"model: {MODEL_NAME}\nweights: {WEIGHTS.name}\n"
                f"eval(KLUE-NER validation): {len(val_ds)}건\n\n"
                f"entity F1: {f1:.4f}  precision: {p:.4f}  recall: {r:.4f}\n\n{report}\n")
            mlflow.log_artifact(str(rp))

    print(f"[logged] klue-bert-base-ner: F1={f1:.4f} P={p:.4f} R={r:.4f}")
    print("완료 → MLflow UI: http://127.0.0.1:5000 (실험: ner-klue)")


if __name__ == "__main__":
    main()
