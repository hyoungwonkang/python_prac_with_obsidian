"""
지시 2 '분류 잘하는 법' — 고정 test셋(300건) 하나로 여러 분류 방법을 동일 잣대 비교.

방법:
  - RULE: rule_spam.py (키워드+패턴, train으로만 제작)
  - BERT: 산출물 3종 세트 폴더들 (지시 1 환경 artifacts/ 재사용)

기록: MLflow sqlite(이 폴더 mlflow.db), 실험 '분류방법-비교' — 방법 1개 = run 1개.
사용:
  ~/rnd-env/bin/python eval_compare.py
  ARTIFACTS=경로1,경로2 ~/rnd-env/bin/python eval_compare.py   # BERT 산출물 지정(콤마 구분)
"""
import json
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rule_spam import THRESHOLD, classify

HERE = Path(__file__).resolve().parent
TEST = Path(os.environ.get("TEST", HERE / "../../rnd-bert-labeling-test/export/ko/test.csv")).resolve()
ARTIFACTS = os.environ.get(
    "ARTIFACTS", str(HERE / "../../rnd-dataset-artifacts/export/artifacts/ko-spam-smoke"))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def four_cells(golds, preds):
    """혼동행렬 네 칸 — 정탐(tp)·오탐(fp)·미탐(fn)·tn."""
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 0)
    return tp, fp, fn, tn


def eval_bert(artifact_dir, texts):
    """산출물 3종 세트(가중치+label_map+meta)만으로 분류 — 학습 코드 불필요."""
    art = Path(artifact_dir).resolve()
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    label_map = json.loads((art / "label_map.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(meta["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=meta["num_labels"])
    model.load_state_dict(torch.load(art / "model.pt", map_location="cpu"))
    model.to(DEVICE).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, max_length=meta["max_len"],
                            padding=True, return_tensors="pt").to(DEVICE)
            ids = model(**enc).logits.argmax(-1).cpu().tolist()
            preds += [int(id2label[i_]) for i_ in ids]   # 클래스명 '0'/'1' → 정수 라벨
    return preds, meta


def report(name, golds, preds, params):
    tp, fp, fn, tn = four_cells(golds, preds)
    metrics = {
        "정확도": accuracy_score(golds, preds),
        "스팸정밀도": precision_score(golds, preds, zero_division=0),
        "스팸재현율": recall_score(golds, preds, zero_division=0),
        "스팸F1": f1_score(golds, preds, zero_division=0),
        "정탐tp": tp, "오탐fp": fp, "미탐fn": fn, "tn": tn,
    }
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print(f"{name:<16} 정확도 {metrics['정확도']:.4f} | P {metrics['스팸정밀도']:.4f} "
          f"R {metrics['스팸재현율']:.4f} F1 {metrics['스팸F1']:.4f} | "
          f"정탐 {tp} 오탐 {fp} 미탐 {fn}")
    return metrics


def main():
    df = pd.read_csv(TEST)
    golds = df["Label"].tolist()
    texts = df["Text"].astype(str).tolist()
    print(f"고정 test셋: {TEST.name} {len(df)}건 (스팸 {sum(golds)} / 정상 {len(golds) - sum(golds)})")

    # 기본은 로컬 sqlite(과정 기록). 주간보고용 최종 지표는 통합 서버에:
    #   MLFLOW_URI=http://127.0.0.1:5000 ~/rnd-env/bin/python eval_compare.py
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI", f"sqlite:///{HERE / 'mlflow.db'}"))
    mlflow.set_experiment("분류방법-비교")

    t0 = time.time()
    preds = [classify(t) for t in texts]
    report("RULE", golds, preds,
           {"방법": "키워드+패턴", "문턱": THRESHOLD, "제작데이터": "train만(누출금지)"})
    print(f"  (RULE 소요 {time.time() - t0:.2f}초)")

    for art in [a.strip() for a in ARTIFACTS.split(",") if a.strip()]:
        t0 = time.time()
        preds, meta = eval_bert(art, texts)
        report(f"BERT-{Path(art).name}", golds, preds,
               {"방법": "BERT 산출물 재사용", "산출물": Path(art).name,
                "베이스": meta["base_model"], "학습기록": json.dumps(meta["metrics"], ensure_ascii=False)})
        print(f"  (BERT 소요 {time.time() - t0:.1f}초)")


if __name__ == "__main__":
    main()
