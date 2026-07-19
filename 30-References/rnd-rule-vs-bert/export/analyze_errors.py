"""
'분류 잘하는 법' — 2차 실험: 오답 정성 분석 + 하이브리드 결합.

재료는 고정 test셋 300건과 기존 예측뿐 — 재학습 없음.
  1) 오답 정성 분석: RULE과 BERT의 미탐·오탐이 같은 문장인지(교집합) 확인
     → 겹치면 데이터 한계(둘 다 못 푸는 문제), 다르면 결합(앙상블)의 여지
  2) 하이브리드: OR(룰이 정상이라 한 것만 BERT가 2차 판정과 동치) / AND(둘 다 스팸일 때만)

사용:
  python analyze_errors.py
"""
import os
from pathlib import Path

import mlflow
import pandas as pd

from eval_compare import eval_bert, four_cells, report
from rule_spam import classify

HERE = Path(__file__).resolve().parent
TEST = Path(os.environ.get("TEST", HERE / "../../rnd-bert-labeling-test/export/ko/test.csv")).resolve()
BERT_ART = os.environ.get(
    "ARTIFACT", str(HERE / "../../rnd-dataset-artifacts/export/artifacts/ko-spam-full"))


def show(title, rows, limit=10):
    print(f"\n── {title} ({len(rows)}건) " + "─" * 30)
    for text in rows[:limit]:
        t = str(text).replace("\n", " ")
        print(f"  · {t[:70]}{'…' if len(t) > 70 else ''}")
    if len(rows) > limit:
        print(f"  … 외 {len(rows) - limit}건")


def main():
    df = pd.read_csv(TEST)
    golds = df["Label"].tolist()
    texts = df["Text"].astype(str).tolist()

    rule_preds = [classify(t) for t in texts]
    bert_preds, _ = eval_bert(BERT_ART, texts)

    # ---- 1) 오답 정성 분석: 미탐(실제 스팸인데 정상판정)·오탐(실제 정상인데 스팸판정)의 겹침 ----
    rule_fn = {i for i, (p, g) in enumerate(zip(rule_preds, golds)) if p == 0 and g == 1}
    bert_fn = {i for i, (p, g) in enumerate(zip(bert_preds, golds)) if p == 0 and g == 1}
    rule_fp = {i for i, (p, g) in enumerate(zip(rule_preds, golds)) if p == 1 and g == 0}
    bert_fp = {i for i, (p, g) in enumerate(zip(bert_preds, golds)) if p == 1 and g == 0}

    print(f"고정 test셋 {len(df)}건 — RULE 미탐 {len(rule_fn)} / BERT 미탐 {len(bert_fn)} "
          f"/ 둘 다 미탐(교집합) {len(rule_fn & bert_fn)}")
    print(f"오탐 — RULE {len(rule_fp)} / BERT {len(bert_fp)} / 교집합 {len(rule_fp & bert_fp)}")

    show("둘 다 놓친 스팸 (데이터·방법 공통의 한계 후보)", [texts[i] for i in sorted(rule_fn & bert_fn)])
    show("RULE만 놓친 스팸 (BERT는 잡음 → 문맥 일반화의 몫)", [texts[i] for i in sorted(rule_fn - bert_fn)])
    show("BERT만 놓친 스팸 (RULE은 잡음 → 키워드의 몫)", [texts[i] for i in sorted(bert_fn - rule_fn)])
    show("RULE 오탐 (억울한 정상)", [texts[i] for i in sorted(rule_fp)])
    show("BERT 오탐 (억울한 정상)", [texts[i] for i in sorted(bert_fp)])

    # ---- 2) 하이브리드 결합 ----
    or_preds = [1 if r == 1 or b == 1 else 0 for r, b in zip(rule_preds, bert_preds)]
    and_preds = [1 if r == 1 and b == 1 else 0 for r, b in zip(rule_preds, bert_preds)]

    print("\n── 하이브리드 채점 (MLflow '분류방법-비교'에 기록) " + "─" * 20)
    # 기본은 로컬 sqlite(과정 기록). 통합 서버 기록: MLFLOW_URI=http://127.0.0.1:5000
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI", f"sqlite:///{HERE / 'mlflow.db'}"))
    mlflow.set_experiment("분류방법-비교")
    report("HYBRID-OR", golds, or_preds,
           {"방법": "룰 OR BERT (룰 정상판정분만 BERT 2차와 동치)", "구성": "RULE+BERT-full"})
    report("HYBRID-AND", golds, and_preds,
           {"방법": "룰 AND BERT (둘 다 스팸일 때만)", "구성": "RULE+BERT-full"})


if __name__ == "__main__":
    main()
