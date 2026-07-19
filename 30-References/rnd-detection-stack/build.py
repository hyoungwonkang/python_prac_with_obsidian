"""
보고 패키지 조립기 — 통합 문서(이 폴더) + 각 R&D 코드를 합쳐 상사에게 보낼 폴더를 만든다.

레거시 99-Archive/rnd-report-package/build.py의 후속 (rnd-detection-stack 로드맵 Step 5):
  - 산출물 폴더에 날짜 포함 (발송본 불명 재발 방지 — 보낸 스냅샷은 99-Archive에 보관)
  - 경량본이 기본 (WEIGHTS=1 일 때만 가중치 포함) — 상사는 매번 새 파일을 받는다는 전제
  - 스팸 재생성용 csv가 없으면 조용히 건너뛰지 않고 중단 (경량본의 재생성 절차가 무너지므로)

산출: <OUT>/  (기본 30-References/탐지분석계층-보고-<오늘날짜> — vault 안에서 내용 확인 후 zip·발송)
  - 최상위: README 2종·1~6 문서·requirements.txt (이 폴더에서 복사 — 통합본, 로드맵 제외)
  - rnd-*/export/: 코드만 (*.py·*.json·*.yaml·*.txt) — 개별 문서·가중치·데이터·캐시 제외
  ※ 폴더 나열 구조를 유지해야 데모(app.py)·ocr_pipeline.py의 상대경로 참조가 작동한다.

실행:  python build.py           또는   OUT=/보낼/경로 WEIGHTS=1 python build.py
"""
import datetime
import os
import re
import shutil
from pathlib import Path

FRONT = Path(__file__).resolve().parent          # rnd-detection-stack (통합 문서)
REFS = FRONT.parent                              # 30-References
TODAY = datetime.date.today().isoformat()
OUT = Path(os.environ.get("OUT", REFS / f"탐지분석계층-보고-{TODAY}")).expanduser()

FOLDERS = ["rnd-dataset-artifacts", "rnd-rule-vs-bert", "rnd-clip",
           "rnd-uxui-demo", "rnd-detection-models", "rnd-ocr"]
FRONT_FILES = ["README_mac.md", "README_windows.md", "1_연구문서.md", "2_소스코드.md",
               "3_사용법.md", "4_가이드.md", "5_도식도.md", "6_보고서.md", "requirements.txt"]
FRONT_PDFS = ["5_도식도.pdf", "6_보고서.pdf"]   # make_pdf.py로 생성 (있으면 포함)
KEEP = {".py", ".json", ".yaml", ".txt"}         # 코드·소형 데이터만
SKIP = {"requirements.txt"}                       # 개별 req는 통합본으로 대체 (문서 .md는 KEEP 밖이라 자동 제외)

# 경량본이 기본 — WEIGHTS=1 일 때만 가중치 포함 (재생성 절차는 README §2)
INCLUDE_WEIGHTS = os.environ.get("WEIGHTS", "0") == "1"
WEIGHT_ITEMS = [
    "rnd-dataset-artifacts/export/artifacts/ko-spam-full/model.pt",       # 스팸 분류
    "rnd-dataset-artifacts/export/artifacts/ko-spam-full/label_map.json",
    "rnd-dataset-artifacts/export/artifacts/ko-spam-full/meta.json",
    "rnd-detection-models/export/ner_klue.pt",                            # NER
    "rnd-detection-models/export/yolov8n.pt",                             # YOLO 기본 (없으면 자동 다운로드)
]
# 스팸 가중치 재생성용 학습 데이터(공개 SMS 스팸, ~210KB) — 경량본 재생성 절차의 필수 재료
DATA_ITEMS = [
    ("rnd-bert-labeling-test/export/ko/train.csv", "rnd-dataset-artifacts/export/datasets/ko-spam/train.csv"),
    ("rnd-bert-labeling-test/export/ko/validation.csv", "rnd-dataset-artifacts/export/datasets/ko-spam/validation.csv"),
    ("rnd-bert-labeling-test/export/ko/test.csv", "rnd-dataset-artifacts/export/datasets/ko-spam/test.csv"),
]


def main():
    missing_csv = [s for s, _ in DATA_ITEMS if not (REFS / s).exists()]
    if missing_csv:
        raise SystemExit(
            "⚠️ 스팸 재생성용 csv 없음 — 경량본의 가중치 재생성 절차가 무너지므로 조립 중단.\n"
            "   생성:  cd 30-References/rnd-bert-labeling-test/export && python dataset_finetuning_ko.py\n"
            "   누락: " + ", ".join(missing_csv))

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
            raise SystemExit(f"⚠️ {folder}/export 없음 — 모듈 누락 상태로는 조립하지 않는다.")
        dst = OUT / folder / "export"
        dst.mkdir(parents=True)
        n = 0
        for f in sorted(src.iterdir()):           # 최상위 파일만 (하위 datasets·artifacts·mlruns 등 디렉터리는 제외)
            if f.is_file() and f.suffix in KEEP and f.name not in SKIP:
                shutil.copy2(f, dst / f.name)
                n += 1
        print(f"  {folder}/export → 코드 {n}개")
        total += n

    for src_rel, dst_rel in DATA_ITEMS:        # 스팸 재생성용 데이터 (필수 동봉 — 위에서 존재 검증됨)
        dst = OUT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REFS / src_rel, dst)
    print(f"스팸 재생성 데이터 {len(DATA_ITEMS)}개 동봉")

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
        print(f"가중치 {w}개 포함 (WEIGHTS=1 — 바로 실행 가능)")
    else:
        print("경량본 (기본) — 가중치 미포함, 받는 쪽에서 README §2대로 재생성")

    # 로컬 절대경로 세정 — 생성 파일(meta.json 등)에 박힌 작업 경로를 상대형으로 (경로 노출 방지)
    # 이 PC 경로만이 아니라 다른 PC(맥 /Users/…, 윈도우 C:\…)에서 생성돼 박힌 경로도 vault 이름 기준으로 제거
    VAULT_RE = re.compile(r"[^\"'\s]*python_prac_with_obsidian[/\\]+")
    scrubbed = 0
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".txt", ".md", ".yaml"):
            t = p.read_text(encoding="utf-8")
            t2 = VAULT_RE.sub("", t)
            if t2 != t:
                p.write_text(t2, encoding="utf-8")
                scrubbed += 1
    if scrubbed:
        print(f"경로 세정 {scrubbed}개 파일 (로컬 절대경로 제거)")

    print(f"\n조립 완료: {OUT}  (코드 {total}개" + (f" + 가중치 {w}개)" if INCLUDE_WEIGHTS else ", 경량본)"))
    print("자동 제외: 개인 사진·실험DB·캐시·개별 문서. PII·이미지 검색 탭은 가중치 없이도 동작.")
    # zip은 python zipfile로 — Info-ZIP(zip -r)산 한글명 zip은 윈도우 내장 해제기가 못 연다 (2026-07-19 실측)
    print(f"압축:  cd {OUT.parent} && python -m zipfile -c '{OUT.name}.zip' '{OUT.name}'")
    print(f"발송 후:  스냅샷을 99-Archive/{OUT.name}/ 로 보관 (발송본 추적)")


if __name__ == "__main__":
    main()
