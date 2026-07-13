"""
보고 패키지 조립기 — 통합 문서(이 폴더) + 각 R&D 코드를 합쳐 상사에게 보낼 폴더를 만든다.

산출: <OUT>/  (기본 ~/Desktop/탐지계층-R&D-보고)
  - 최상위: README·1~4 문서·requirements.txt (이 폴더에서 복사 — 통합본)
  - rnd-*/export/: 코드만 (*.py·*.json·*.yaml·*.txt) — 개별 문서·req·README·가중치·데이터·캐시 제외
  ※ 폴더 나열 구조를 유지해야 데모(app.py)의 상대경로 참조가 작동한다.

실행:  python build.py           또는   OUT=/보낼/경로 python build.py
"""
import os
import shutil
from pathlib import Path

FRONT = Path(__file__).resolve().parent          # rnd-report-package (통합 문서)
REFS = FRONT.parent                              # 30-References
OUT = Path(os.environ.get("OUT", "~/Desktop/탐지계층-R&D-보고")).expanduser()

FOLDERS = ["rnd-dataset-artifacts", "rnd-rule-vs-bert", "rnd-clip",
           "rnd-uxui-demo", "rnd-detection-models"]
FRONT_FILES = ["README.md", "1_연구문서.md", "2_소스코드.md", "3_사용법.md",
               "4_가이드.md", "5_도식도.md", "6_보고서.md", "requirements.txt"]
FRONT_PDFS = ["5_도식도.pdf", "6_보고서.pdf"]   # make_pdf.py로 생성 (있으면 포함)
KEEP = {".py", ".json", ".yaml", ".txt"}         # 코드·소형 데이터만
SKIP = {"requirements.txt"}                       # 개별 req는 통합본으로 대체 (문서 .md는 KEEP 밖이라 자동 제외)

# 데모 구동에 필요한 가중치 (WEIGHTS=0 이면 미포함 — 가벼운 배포용). 재생성엔 학습 데이터도 필요하므로 기본 포함.
INCLUDE_WEIGHTS = os.environ.get("WEIGHTS", "1") != "0"
WEIGHT_ITEMS = [
    "rnd-dataset-artifacts/export/artifacts/ko-spam-full/model.pt",       # 스팸 분류
    "rnd-dataset-artifacts/export/artifacts/ko-spam-full/label_map.json",
    "rnd-dataset-artifacts/export/artifacts/ko-spam-full/meta.json",
    "rnd-detection-models/export/ner_klue.pt",                            # NER
    "rnd-detection-models/export/yolov8n.pt",                             # YOLO 기본
]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for f in FRONT_FILES:
        shutil.copy2(FRONT / f, OUT / f)
    n_pdf = 0
    for f in FRONT_PDFS:
        if (FRONT / f).exists():
            shutil.copy2(FRONT / f, OUT / f)
            n_pdf += 1
    print(f"통합 문서 {len(FRONT_FILES)}종 + PDF {n_pdf}개 → {OUT.name}/ 최상위"
          + ("" if n_pdf == len(FRONT_PDFS) else "  ⚠️ PDF 누락 — make_pdf.py 먼저 실행"))

    total = 0
    for folder in FOLDERS:
        src = REFS / folder / "export"
        if not src.is_dir():
            print(f"  ⚠️ {folder}/export 없음 — 건너뜀")
            continue
        dst = OUT / folder / "export"
        dst.mkdir(parents=True)
        n = 0
        for f in sorted(src.iterdir()):           # 최상위 파일만 (하위 datasets·artifacts·mlruns 등 디렉터리는 제외)
            if f.is_file() and f.suffix in KEEP and f.name not in SKIP:
                shutil.copy2(f, dst / f.name)
                n += 1
        print(f"  {folder}/export → 코드 {n}개")
        total += n

    w = 0
    if INCLUDE_WEIGHTS:
        for rel in WEIGHT_ITEMS:
            src = REFS / rel
            if src.exists():
                dst = OUT / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                w += 1
            else:
                print(f"  ⚠️ 가중치 없음(원본): {rel} — 데모에서 해당 탭은 안내 메시지 표시")
        print(f"가중치 {w}개 포함 (바로 실행 가능)")
    else:
        print("가중치 미포함 (WEIGHTS=0) — 받는 쪽에서 재생성 필요")

    print(f"\n조립 완료: {OUT}  (코드 {total}개" + (f" + 가중치 {w}개)" if INCLUDE_WEIGHTS else ")"))
    print("자동 제외: 개인 사진·실험DB·캐시·개별 문서. "
          "PII·이미지 검색 탭은 가중치 없이도 동작.")
    print(f"압축:  cd {OUT.parent} && zip -r '{OUT.name}.zip' '{OUT.name}'")


if __name__ == "__main__":
    main()
