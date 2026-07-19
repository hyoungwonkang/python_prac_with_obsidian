"""
PII 분류 체계(라벨 스키마) 문서 생성기 — ko-pii 내장 33라벨을 기계가 읽는 정본으로 추출.

ko_pii.labels의 ALL_LABELS·LABEL_INFO·GROUPS를 읽어
  pii_label_schema.md   (사람용 표 — 라벨/한국어명/그룹/검증방식/샘플 커버 여부)
  pii_label_schema.json (기계용 — 검증기·후속 도구가 참조)
를 생성한다. 평가 데이터(pii_samples.txt)가 33라벨 중 몇 개를 커버하는지 함께 표시해
데이터셋을 어느 방향으로 늘려야 하는지 보이게 한다.

실행:
  ~/rnd-env/bin/python make_pii_schema.py
  SAMPLES=/path/to/pii_samples.txt ~/rnd-env/bin/python make_pii_schema.py
"""
import json
import os
from importlib.metadata import version
from pathlib import Path

from ko_pii.labels import ALL_LABELS, LABEL_INFO

HERE = Path(__file__).resolve().parent
SAMPLES = Path(os.environ.get(
    "SAMPLES", HERE / "../../rnd-detection-models/export/pii_samples.txt")).resolve()
OUT_MD = HERE / "pii_label_schema.md"
OUT_JSON = HERE / "pii_label_schema.json"


def load_covered_labels():
    """평가 샘플 파일에서 기대 라벨 집합을 수집 (파일 없으면 빈 집합)."""
    covered = set()
    if not SAMPLES.exists():
        print(f"⚠️ 샘플 파일 없음: {SAMPLES} — 커버리지 표시 생략")
        return covered
    for line in SAMPLES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        _, expected = line.split("\t", 1)
        covered.update(e.strip() for e in expected.split(",") if e.strip())
    return covered


def main():
    covered = load_covered_labels()
    ko_pii_ver = version("ko-pii")

    # 그룹 순서: LABEL_INFO 등장 순서를 유지
    group_order = []
    for label in LABEL_INFO:
        _, group, _ = LABEL_INFO[label]
        if group not in group_order:
            group_order.append(group)

    # ---- JSON (기계용 정본) ----
    schema = {
        "source": f"ko-pii {ko_pii_ver} (ko_pii.labels)",
        "total_labels": len(ALL_LABELS),
        "groups": group_order,
        "labels": {
            label: {
                "korean": LABEL_INFO[label][0],
                "group": LABEL_INFO[label][1],
                "method": LABEL_INFO[label][2],
                "covered_by_samples": label in covered,
            }
            for label in sorted(ALL_LABELS)
        },
        "samples_file": str(SAMPLES),
        "coverage": {"covered": len(covered & set(ALL_LABELS)),
                     "total": len(ALL_LABELS)},
    }
    OUT_JSON.write_text(json.dumps(schema, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- Markdown (사람용 표) ----
    cov_n, total = schema["coverage"]["covered"], schema["coverage"]["total"]
    lines = [
        "# PII 분류 체계 (라벨 스키마)",
        "",
        f"- 출처: **ko-pii {ko_pii_ver}** 내장 라벨 정의(`ko_pii.labels`) — 코드에서 자동 추출 (`make_pii_schema.py`)",
        f"- 전체 **{total}개 라벨**, 탐지 방식 **{len(group_order)}개 그룹**",
        f"- 평가 데이터(`pii_samples.txt`) 커버리지: **{cov_n}/{total}** "
        f"(✅=샘플에 기대 라벨로 등장, ▫️=미커버 → 샘플 확장 후보)",
        "",
    ]
    for group in group_order:
        members = [l for l in LABEL_INFO if LABEL_INFO[l][1] == group]
        lines += [f"## {group} ({len(members)}개)", "",
                  "| 커버 | 라벨 | 한국어명 | 검증 방식 |", "|---|---|---|---|"]
        for label in members:
            korean, _, method = LABEL_INFO[label]
            mark = "✅" if label in covered else "▫️"
            lines.append(f"| {mark} | `{label}` | {korean} | {method} |")
        lines.append("")
    lines += [
        "---",
        "",
        "## 사용 규칙",
        "",
        "- 평가 데이터(`pii_samples.txt`)의 기대 라벨은 **반드시 위 33개 중 하나** — "
        "`validate_pii_samples.py`가 강제한다.",
        "- 새 샘플을 추가할 때는 ▫️(미커버) 라벨을 우선 채워 커버리지를 넓힌다.",
        "- 이 문서는 손으로 고치지 않는다 — ko-pii 업그레이드 시 "
        "`make_pii_schema.py` 재실행으로 갱신.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"생성 완료 → {OUT_MD.name}, {OUT_JSON.name}")
    print(f"라벨 {total}개 / 그룹 {len(group_order)}개 / 샘플 커버리지 {cov_n}/{total}")


if __name__ == "__main__":
    main()
