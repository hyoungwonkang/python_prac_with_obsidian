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
               "4_가이드.md", "requirements.txt"]
KEEP = {".py", ".json", ".yaml", ".txt"}         # 코드·소형 데이터만
SKIP = {"requirements.txt"}                       # 개별 req는 통합본으로 대체 (문서 .md는 KEEP 밖이라 자동 제외)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for f in FRONT_FILES:
        shutil.copy2(FRONT / f, OUT / f)
    print(f"통합 문서 {len(FRONT_FILES)}종 → {OUT.name}/ 최상위")

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

    print(f"\n조립 완료: {OUT}  (코드 총 {total}개 파일)")
    print("보내기 전: ① 무거운 가중치는 미포함 — 받는 쪽에서 재생성(README 1단계) "
          "② 개인 사진·실험DB·캐시는 자동 제외됨")
    print(f"압축:  cd {OUT.parent} && zip -r '{OUT.name}.zip' '{OUT.name}'")


if __name__ == "__main__":
    main()
