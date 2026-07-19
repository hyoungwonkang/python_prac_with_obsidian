"""
PII 평가 데이터 라벨 검증기 — 샘플 파일의 기대 라벨이 ko-pii 분류 체계(33라벨)에
전부 속하는지 확인한다. 오타·비표준 라벨이 데이터에 섞이는 것을 원천 차단(데이터 규약 강제).

검사 항목:
  1. 형식 — 각 데이터 줄은 `문장<TAB>라벨1,라벨2` 형태여야 함
  2. 라벨 유효성 — 모든 라벨이 ko_pii.labels.ALL_LABELS의 구성원이어야 함
  3. 빈 라벨 — 라벨 칸이 비면 오류

통과 시 종료코드 0, 위반 발견 시 1 (CI·훅에 연결 가능).

실행:
  ~/rnd-env/bin/python validate_pii_samples.py
  SAMPLES=/path/to/pii_samples.txt ~/rnd-env/bin/python validate_pii_samples.py
"""
import os
import sys
from pathlib import Path

from ko_pii.labels import ALL_LABELS

HERE = Path(__file__).resolve().parent
SAMPLES = Path(os.environ.get(
    "SAMPLES", HERE / "../../rnd-detection-models/export/pii_samples.txt")).resolve()


def main():
    if not SAMPLES.exists():
        print(f"❌ 샘플 파일 없음: {SAMPLES}")
        sys.exit(1)

    valid_labels = set(ALL_LABELS)
    errors = []
    n_rows = 0
    seen_labels = set()

    for lineno, raw in enumerate(SAMPLES.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            errors.append(f"  {lineno}행: TAB 구분 없음 — `문장<TAB>라벨` 형식 위반: {line[:40]}")
            continue
        text, expected = line.split("\t", 1)
        labels = [e.strip() for e in expected.split(",")]
        if not text.strip():
            errors.append(f"  {lineno}행: 문장이 비어 있음")
        if not any(labels):
            errors.append(f"  {lineno}행: 라벨이 비어 있음")
            continue
        for label in labels:
            if label and label not in valid_labels:
                errors.append(f"  {lineno}행: 비표준 라벨 `{label}` — "
                              f"ko-pii {len(valid_labels)}라벨에 없음 "
                              f"(pii_label_schema.md 참조)")
        seen_labels.update(l for l in labels if l in valid_labels)
        n_rows += 1

    print(f"검증 대상: {SAMPLES}")
    print(f"데이터 {n_rows}건 / 사용 라벨 {len(seen_labels)}종 "
          f"(전체 체계 {len(valid_labels)}라벨 중)")
    if errors:
        print(f"\n❌ 위반 {len(errors)}건:")
        print("\n".join(errors))
        sys.exit(1)
    print("✅ 모든 라벨이 분류 체계에 부합")


if __name__ == "__main__":
    main()
