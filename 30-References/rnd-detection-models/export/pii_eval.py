"""
PII 커버리지 "테스트" — ko-pii가 샘플 문장의 기대 PII 유형을 얼마나 잡는지 측정.

학습 모델이 아니므로 정확도/F1 대신, 손수 라벨한 소량 샘플(pii_samples.txt)에서
**기대 유형 대비 탐지 커버리지(재현율 성격)**를 확인한다. 룰+체크섬 기반의 강점
(주민번호·카드 등 형식/체크섬 검증)과 한계(문맥 의존 개체는 놓칠 수 있음)를 드러낸다.

실행:
  python pii_eval.py
"""
from pathlib import Path

from ko_pii import Anonymizer, ProcessingMode

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "pii_samples.txt"


def load_samples():
    rows = []
    for line in SAMPLES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        text, expected = line.split("\t", 1)
        rows.append((text.strip(), [e.strip() for e in expected.split(",")]))
    return rows


def main():
    anonymizer = Anonymizer(mode=ProcessingMode.STRICT, strategy="tokenize")
    rows = load_samples()

    total_expected = 0
    total_hit = 0
    print(f"샘플 {len(rows)}건 커버리지 평가\n" + "=" * 70)
    for text, expected in rows:
        result = anonymizer.process(text)
        found = {item.detection.label for item in result.detections}
        hits = [e for e in expected if e in found]
        misses = [e for e in expected if e not in found]
        total_expected += len(expected)
        total_hit += len(hits)
        mark = "✅" if not misses else "⚠️"
        print(f"{mark} {text[:40]}")
        print(f"    기대: {expected}  탐지: {sorted(found)}"
              + (f"  누락: {misses}" if misses else ""))

    cov = total_hit / total_expected if total_expected else 0.0
    print("=" * 70)
    print(f"커버리지: {total_hit}/{total_expected} = {cov:.1%} "
          f"(기대 PII 유형 중 탐지된 비율)")


if __name__ == "__main__":
    main()
