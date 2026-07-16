"""
미니 파이프라인 — 이미지 → ②OCR 추출 → ③1차 탐지(스팸 분류·NER·PII) 연결 실증.

"OCR 출력이 BERT/NER 입력으로 이어지는" 요건의 실물. 같은 이미지에서 나온
텍스트 소스 3종을 같은 하류 모듈에 넣어, OCR 품질이 탐지를 어떻게 좌우하는지 비교:
  정답(ground truth — 상한선) / EasyOCR / PaddleOCR
OCR은 엔진별 전용 환경(easy=rnd-env, paddle=ocr-env)의 **서브프로세스**로 실행하고
텍스트 파일(out/)로 인계 — 실제 아키텍처의 계층 분리(②추출 ↔ ③탐지)와 같은 구조.
하류 3종은 기존 R&D 산출물 재사용 (신규 학습 0):
  스팸 = rnd-dataset-artifacts 3종 세트 / NER = predict_ner 함수+ner_klue.pt / PII = ko-pii

실행:  ~/rnd-env/bin/python ocr_pipeline.py          (IMAGE=images/clean.png 기본)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
REFS = HERE.parent.parent                     # 30-References
IMAGE = HERE / os.environ.get("IMAGE", "images/clean.png")
OUT = HERE / "out"
PY = {"easy": Path.home() / "rnd-env/bin/python",
      "paddle": Path.home() / "ocr-env/bin/python"}


# ---------- ② OCR 추출 — 엔진별 전용 환경에서 서브프로세스, 파일로 인계 ----------
def ocr_text(engine: str) -> str:
    """ocr_eval.make_reader 재사용. stdout은 Paddle 로그로 오염되므로 파일 인계."""
    out = OUT / f"ocr_{engine}.txt"
    code = ("import pathlib, ocr_eval\n"
            f"t = ocr_eval.make_reader()(pathlib.Path(r'{IMAGE}'))\n"
            f"pathlib.Path(r'{out}').write_text(t, encoding='utf-8')\n")
    r = subprocess.run([str(PY[engine]), "-c", code], cwd=HERE,
                       capture_output=True, text=True,
                       env={**os.environ, "ENGINE": engine})
    if r.returncode != 0:
        raise SystemExit(f"{engine} OCR 실패:\n{r.stderr[-800:]}")
    return " ".join(out.read_text(encoding="utf-8").split())


# ---------- ③-1 스팸 분류 — 산출물 3종 세트 재사용 (공용 산출물 규약) ----------
def load_spam():
    art = REFS / "rnd-dataset-artifacts/export/artifacts/ko-spam-full"
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in json.loads(
        (art / "label_map.json").read_text(encoding="utf-8"))["id2label"].items()}
    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    mdl = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=meta["num_labels"])
    mdl.load_state_dict(torch.load(art / "model.pt", map_location="cpu"))
    mdl.eval()

    def classify(text):
        enc = tok(text, truncation=True, max_length=meta["max_len"], return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits, -1)[0]
        return id2label[int(p.argmax())], float(p.max())
    return classify


# ---------- ③-2 NER — 기존 추론기의 load/tag_sentence 함수 재사용 ----------
def load_ner():
    sys.path.insert(0, str(REFS / "rnd-detection-models/export"))
    import predict_ner as P
    tok, mdl, id2label = P.load()
    return lambda text: P.tag_sentence(text, tok, mdl, id2label)


# ---------- ③-3 PII — ko-pii 직접 (pii_detect.py와 동일 설정) ----------
def load_pii():
    from ko_pii import Anonymizer, ProcessingMode
    anon = Anonymizer(mode=ProcessingMode.STRICT, strategy="tokenize")

    def detect(text):
        res = anon.process(text)
        dets = [(i.detection.label, i.detection.text) for i in res.detections]
        return dets, res.summary.get("combined_risk")
    return detect


def main():
    OUT.mkdir(exist_ok=True)
    gold = " ".join((IMAGE.parent / "ground_truth.txt").read_text(encoding="utf-8").split())

    print(f"이미지: {IMAGE.name} — ② 추출 중 (easy=rnd-env / paddle=ocr-env 서브프로세스)")
    sources = {"정답": gold, "EasyOCR": ocr_text("easy"), "PaddleOCR": ocr_text("paddle")}

    print("③ 하류 모듈 로드 (기존 산출물 재사용)")
    classify, ner, pii = load_spam(), load_ner(), load_pii()

    rows = []
    for name, text in sources.items():
        label, prob = classify(text)
        entities = ner(text)
        dets, risk = pii(text)
        phone_ok = any("PHONE" in lb.upper() or "MOBILE" in lb.upper() for lb, _ in dets)
        rows.append((name, text, label, prob, entities, dets, risk, phone_ok))

    print("\n" + "=" * 78)
    for name, text, label, prob, entities, dets, risk, phone_ok in rows:
        print(f"[{name}] {text}")
        print(f"  스팸 분류: [{label}] {prob:.2%}")
        print(f"  NER      : {', '.join(f'{t}[{y}]' for t, y in entities) or '—'}")
        print(f"  PII      : {len(dets)}건 " +
              ", ".join(f"[{lb}]{tx}" for lb, tx in dets) + f" | 종합위험도 {risk}")
        print("-" * 78)

    print(f"{'소스':<10} {'스팸판정':<14} {'NER개체':>4}  {'PII':>3}  전화번호PII")
    for name, _, label, prob, entities, dets, _, phone_ok in rows:
        print(f"{name:<10} {label}({prob:.0%})     {len(entities):>4}  {len(dets):>3}  "
              f"{'✅' if phone_ok else '❌ 미탐'}")


if __name__ == "__main__":
    main()
